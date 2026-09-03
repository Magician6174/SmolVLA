"""Frozen SmolVLM2 backbone: the "VL" of the VLA.

WHAT THIS FILE IS FOR
---------------------
SmolVLA is a 450M-parameter model of which only ~100M is trained. This file
owns the other ~350M: a pretrained vision-language model that is loaded, cut
down to its first 16 decoder layers, frozen, and used purely as a conditioning
encoder. `model.py` owns the trainable action expert.

The split matters conceptually. The prefix
    [ img tokens (x3 cameras) | language tokens | state token ]
is computed WITHOUT ever seeing the noisy actions or the flow-matching
timestep. That is enforced by the attention mask (see `build_prefix_masks`):
prefix tokens form one bidirectional block, and nothing in it can attend
forward to the action tokens. The consequence is the entire reason inference
is affordable: the prefix keys and values are computed ONCE per observation
and reused for all 10 Euler steps. If the prefix could see the action tokens,
you would pay the full VLM forward 10 times per control step.

WHY A MANUAL LAYER LOOP INSTEAD OF `text_model(...)`
---------------------------------------------------
The action expert cross-attends into the VLM *per layer*: expert layer i reads
the keys and values the VLM produced at layer i. HF's forward returns hidden
states, not per-layer K/V, and re-deriving K/V from hidden states is not
possible (the k_proj/v_proj inputs are the pre-attention layernorm outputs, not
the layer outputs). So we run the Llama block arithmetic ourselves and keep the
K/V on the way past. `parity_test.py` checks this loop reproduces HF's own
`text_model` hidden states to float tolerance, which is the guard against
getting the residual/norm order subtly wrong.

TWO INHERITED QUIRKS, DELIBERATELY KEPT
---------------------------------------
1. sqrt(d) EMBEDDING SCALE. `embed_prefix` in lerobot multiplies image and
   language embeddings by sqrt(960) ~= 31. That is a PaliGemma/openpi
   convention; Llama-family models (which SmolLM2 is) do NOT do this, so the
   frozen VLM is being fed a residual stream ~31x larger than anything it saw
   in pretraining. It is *nearly* harmless because every block starts with
   RMSNorm, which is scale invariant -- but not exactly harmless, because it
   changes the relative magnitude of the attention/MLP updates against the
   residual. Kept on by default so the from-scratch build and the
   `smolvla_base` finetune are comparable; `scale_embeddings=False` to test it.
2. RoPE THETA IS WRONG BY 10x, AND WE KEEP IT WRONG ON PURPOSE. lerobot
   hardcodes `max_wavelength=10_000` in its own `apply_rope` instead of reading
   the checkpoint. SmolVLM2-500M was pretrained with `rope_theta=100_000`
   (verified: `text_config.rope_parameters["rope_theta"] == 100000`; in
   transformers >=5 it is no longer a top-level `rope_theta` attribute, which is
   presumably how the bug survived). So every frozen text layer is being fed
   rotation frequencies 10x higher than the ones its attention heads were fit
   to. Position 0 is unaffected and short sequences are only mildly perturbed,
   but our prefix is ~241 tokens, so the tail of the sequence is meaningfully
   rotated away from anything the pretrained weights ever saw.

   We DEFAULT TO 10_000 anyway: `smolvla_base` was pretrained on 200k steps
   *with* that value, so its expert weights are adapted to the distorted
   conditioning. Using 100_000 for the from-scratch build while the finetuning
   baseline needs 10_000 would confound the one comparison this project exists
   to make. Pass `rope_theta="checkpoint"` to get 100_000 -- that is a planned
   ablation, and it is a fair guess that it is worth a point or two, since it
   costs nothing at inference. `describe()` always prints both values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


LEROBOT_ROPE_THETA = 10_000.0
"""What lerobot's `apply_rope` hardcodes. Not what the checkpoint was trained with."""


def checkpoint_rope_theta(text_cfg) -> float:
    """The rope base the text tower was actually pretrained with.

    transformers moved this from `config.rope_theta` into
    `config.rope_parameters["rope_theta"]` (with `rope_scaling` kept as an
    alias). Reading only the old attribute silently yields a default -- which is
    exactly the failure mode this function exists to prevent.
    """
    for attr in ("rope_parameters", "rope_scaling"):
        params = getattr(text_cfg, attr, None)
        if isinstance(params, dict) and "rope_theta" in params:
            return float(params["rope_theta"])
    theta = getattr(text_cfg, "rope_theta", None)
    if theta is None:
        raise ValueError("could not find rope_theta on the text config")
    return float(theta)


def apply_rope(x: Tensor, positions: Tensor, theta: float = 10_000.0) -> Tensor:
    """Rotary position embedding on x (B, L, H, D) with integer positions (B, L).

    Split-half convention: the first D/2 channels are rotated against the last
    D/2. This is the same rotation HF's Llama does with `rotate_half` plus a
    duplicated cos/sin table, just written without the duplication. Computed in
    float32 regardless of input dtype -- the sin/cos of a large position index
    is exactly where bf16 loses the plot.
    """
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.float()
    exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device)
    timescale = theta**exponents
    radians = positions[..., None].float() / timescale[None, None, :]
    radians = radians[..., None, :]                      # (B, L, 1, D/2) -> broadcasts over heads
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Block-causal 2D attention mask from a 1D "starts a new block" flag.

    `att_masks[i] == 1` means token i begins a new attention block, so earlier
    blocks cannot see it. Taking cumulative sums, token i attends to token j iff
    `cumsum[j] <= cumsum[i]`. Two consequences worth stating out loud because
    they are easy to assume wrong:

      * all-zeros over the image+language tokens  -> ONE bidirectional block
      * all-ones over the 50 action tokens        -> STRICTLY CAUSAL chunk

    So SmolVLA does *not* denoise the action chunk with bidirectional attention:
    action token k only ever sees actions 0..k. That is inherited from openpi
    (pi0) and is a real modelling choice, not an implementation accident.
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"expected 2D masks, got {att_masks.ndim=} {pad_masks.ndim=}")
    cumsum = torch.cumsum(att_masks.to(torch.int64), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d & pad_2d


def ensure_attendable(att_2d: Tensor) -> Tensor:
    """Give every fully-masked query row a self-attention diagonal.

    A padded token's row in the 2D mask is entirely False (`pad_2d` zeroes both
    its row and its column). Softmax over a row of all -inf is 0/0 = NaN, and
    because a NaN value gets multiplied by a zero attention weight rather than
    dropped, one NaN token poisons every token in the next layer. On this
    machine PyTorch 2.10's math backend happens to return zeros instead, which
    is exactly the sort of thing that works on a laptop and produces a loss of
    `nan` on the first CUDA step, where flash/mem-efficient kernels are selected
    and make no such promise.

    Padded rows are discarded downstream (they are masked as KEYS everywhere, so
    nothing ever reads them), which is why handing them a harmless diagonal is
    free. Kept out of `make_att_2d_masks` deliberately: that function has to stay
    bit-identical to lerobot's for `parity_test.py` to mean anything.
    """
    empty = ~att_2d.any(dim=-1, keepdim=True)
    eye = torch.eye(att_2d.shape[-1], dtype=torch.bool, device=att_2d.device)
    return att_2d | (empty & eye)


def resize_with_pad(img: Tensor, height: int, width: int, pad_value: float = 0.0) -> Tensor:
    """Letterbox (b, c, h, w) to (height, width) without changing aspect ratio.

    Our frames are 480x640 (4:3) and SigLIP wants 512x512, so a plain resize
    would squash the scene horizontally by 25% -- a systematic distortion of
    exactly the geometry the policy has to reason about. Padding is applied on
    the LEFT and TOP to match lerobot byte for byte; there is no principled
    reason for that side over any other, but changing it would silently shift
    every position index relative to the pretrained checkpoint.
    """
    if img.ndim != 4:
        raise ValueError(f"(b, c, h, w) expected, got {tuple(img.shape)}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    out = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.pad(out, (max(0, width - new_w), 0, max(0, height - new_h), 0), value=pad_value)


@dataclass
class PrefixCache:
    """Per-layer keys/values of the frozen prefix, plus the mask that produced them.

    keys/values are lists of length `num_layers`, each (B, L, n_kv_heads, head_dim),
    with RoPE already applied to the keys and BEFORE grouped-query expansion
    (storing the expanded form would waste 3x the memory for no benefit).
    """
    keys: list[Tensor]
    values: list[Tensor]
    pad_mask: Tensor          # (B, L) bool, False on padding
    hidden: Tensor            # (B, L, d_vlm) final-normed prefix hidden states
    position_ids: Tensor      # (B, L) int64

    @property
    def length(self) -> int:
        return self.pad_mask.shape[1]

    def to(self, *args, **kwargs) -> PrefixCache:
        return PrefixCache(
            [k.to(*args, **kwargs) for k in self.keys],
            [v.to(*args, **kwargs) for v in self.values],
            self.pad_mask.to(*args, **kwargs),
            self.hidden.to(*args, **kwargs),
            self.position_ids,
        )


class FrozenSmolVLM(nn.Module):
    """SmolVLM2-500M, truncated to `num_layers` decoder layers and frozen."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        num_layers: int = 16,
        scale_embeddings: bool = True,
        rope_theta: float | str = LEROBOT_ROPE_THETA,
        dtype: torch.dtype = torch.float32,
        load_weights: bool = True,
    ):
        """rope_theta: a number, or the string "checkpoint" to use the
        pretrained value (100_000 for SmolVLM2-500M). See the module docstring:
        the default is deliberately lerobot's wrong 10_000.
        """
        super().__init__()
        from transformers import AutoConfig, AutoProcessor, SmolVLMForConditionalGeneration

        if load_weights:
            vlm = SmolVLMForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, low_cpu_mem_usage=True
            )
        else:
            # Random init: only useful for fast shape/plumbing tests.
            vlm = SmolVLMForConditionalGeneration(config=AutoConfig.from_pretrained(model_name))
            vlm = vlm.to(dtype=dtype)

        self.processor = AutoProcessor.from_pretrained(model_name)
        text_cfg = vlm.config.text_config
        total_layers = len(vlm.model.text_model.layers)
        if not 0 < num_layers <= total_layers:
            raise ValueError(f"num_layers must be in 1..{total_layers}, got {num_layers}")
        # Truncating is what makes this a *backbone* rather than a language model:
        # the deleted layers are ~half the parameters and they only ever served
        # next-token prediction, which we never do.
        vlm.model.text_model.layers = vlm.model.text_model.layers[:num_layers]
        vlm.lm_head = None          # never used; dropping it saves ~50M params of VRAM

        self.vlm = vlm
        self.num_layers = num_layers
        self.total_layers = total_layers
        self.hidden_size = text_cfg.hidden_size
        self.num_heads = text_cfg.num_attention_heads
        self.num_kv_heads = getattr(text_cfg, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(text_cfg, "head_dim", None) or self.hidden_size // self.num_heads
        self.config_rope_theta = checkpoint_rope_theta(text_cfg)
        self.rope_theta = (
            self.config_rope_theta if rope_theta == "checkpoint" else float(rope_theta)
        )
        self.scale_embeddings = scale_embeddings
        self.image_size = vlm.config.vision_config.image_size
        self._dtype = dtype

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # `nn.Module.train()` would flip the frozen VLM back into train mode and
    # start updating any BatchNorm-like statistics inside the vision tower.
    def train(self, mode: bool = True) -> FrozenSmolVLM:  # noqa: D102
        return super().train(False)

    @property
    def _text(self):
        return self.vlm.model.text_model

    def describe(self) -> str:
        toks = self.tokens_per_image()
        return (
            f"SmolVLM2 backbone: {self.num_layers}/{self.total_layers} text layers, "
            f"d={self.hidden_size}, heads={self.num_heads} (kv={self.num_kv_heads}), "
            f"head_dim={self.head_dim}, image {self.image_size}px -> {toks} tokens, "
            f"rope_theta={self.rope_theta:g} "
            f"(checkpoint pretrained with {self.config_rope_theta:g}"
            f"{'' if self.rope_theta == self.config_rope_theta else ' -- MISMATCH, see docstring'}), "
            f"scale_embeddings={self.scale_embeddings}, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.1f}M (all frozen)"
        )

    def tokens_per_image(self) -> int:
        """Image tokens after SigLIP patching and SmolVLM's pixel shuffle."""
        patch = self.vlm.config.vision_config.patch_size
        side = self.image_size // patch
        sf = getattr(self.vlm.config, "scale_factor", 1)
        return (side // sf) ** 2

    @torch.no_grad()
    def embed_images(self, pixel_values: Tensor) -> Tensor:
        """(B, 3, H, W) in [-1, 1] -> (B, T_img, d_vlm) token embeddings."""
        vm = self.vlm.model
        feats = vm.vision_model(
            pixel_values=pixel_values.to(dtype=vm.vision_model.dtype), patch_attention_mask=None
        ).last_hidden_state
        # The connector is the pixel-shuffle + MLP that folds 4x4 patch
        # neighbourhoods into one token and maps 768 -> 960.
        emb = vm.connector(feats)
        return emb * math.sqrt(self.hidden_size) if self.scale_embeddings else emb

    @torch.no_grad()
    def embed_text(self, input_ids: Tensor) -> Tensor:
        emb = self._text.get_input_embeddings()(input_ids)
        return emb * math.sqrt(self.hidden_size) if self.scale_embeddings else emb

    # NOT decorated with no_grad: the state token entering this function comes
    # from a TRAINABLE projection, so the graph has to stay intact. The VLM's own
    # parameters are all requires_grad=False, so nothing here is updated -- the
    # only gradient that flows out is the one going back to `state_proj`.
    def encode_prefix(self, embs: Tensor, pad_mask: Tensor, att_mask: Tensor) -> PrefixCache:
        """Run the truncated decoder over the prefix, keeping per-layer K/V.

        embs (B, L, d) are already-embedded tokens -- images, language and the
        state token are concatenated by the caller (`model.py`), because the
        state projection is trainable and therefore does not belong here.
        """
        att_2d = ensure_attendable(make_att_2d_masks(pad_mask, att_mask))[:, None]
        # Position ids skip padding: cumsum-1 over the pad mask means a padded
        # token never advances the RoPE phase for the real tokens after it.
        position_ids = torch.cumsum(pad_mask.to(torch.int64), dim=1) - 1

        h = embs.to(self._dtype)
        keys, values = [], []
        groups = self.num_heads // self.num_kv_heads
        for layer in self._text.layers:
            x = layer.input_layernorm(h)
            b, ln = x.shape[:2]
            q = layer.self_attn.q_proj(x).view(b, ln, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(x).view(b, ln, self.num_kv_heads, self.head_dim)
            v = layer.self_attn.v_proj(x).view(b, ln, self.num_kv_heads, self.head_dim)
            q = apply_rope(q, position_ids, self.rope_theta)
            k = apply_rope(k, position_ids, self.rope_theta)
            keys.append(k)
            values.append(v)

            att = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.repeat_interleave(groups, dim=2).transpose(1, 2),
                v.repeat_interleave(groups, dim=2).transpose(1, 2),
                attn_mask=att_2d,
            )
            att = att.transpose(1, 2).reshape(b, ln, self.num_heads * self.head_dim)
            h = h + layer.self_attn.o_proj(att)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        return PrefixCache(keys, values, pad_mask, self._text.norm(h), position_ids)
