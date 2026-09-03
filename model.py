"""SmolVLA, action expert written from scratch, trained with flow matching.

This is the build where nothing about the policy head is imported. `diffusers`,
lerobot's `SmolVLAPolicy` and its `SmolVLMWithExpertModel` are all absent; the
only borrowed weights are the frozen vision-language backbone in
`vlm_backbone.py`. `model_lib.py` is the counterpart that does the opposite --
library scheduler, lerobot policy, `smolvla_base` weights -- so the two can be
compared. `parity_test.py` checks the pieces of this file against lerobot's
implementation numerically.

FLOW MATCHING IN ONE PARAGRAPH
------------------------------
Pick a straight path from noise to data: x_t = t*eps + (1-t)*A, with t=1 pure
noise and t=0 the true action chunk. Differentiate it: dx/dt = eps - A, a
CONSTANT velocity that does not depend on t at all. Train a network v(x_t, t, c)
to regress that constant with plain MSE. At inference, start at x=eps and
integrate the learned field down to t=0 with Euler steps. Because each
*conditional* path is a straight line, the only integration error comes from
the curvature of the marginal field where paths from different data points
cross, which is why 10 steps suffice here and DDPM needed 100 (or a learned
DDIM schedule to fake it). Full derivation, including why regressing the
conditional velocity gives the same gradient as the intractable marginal
objective, is in FLOW_MATCHING.md.

WHERE THE PARAMETERS ARE
------------------------
    frozen  : SmolVLM2 vision tower + connector + first 16 text layers (~350M)
    trained : state_proj, action_in_proj, time MLP, 16 expert layers,
              action_out_proj                                        (~100M)

ARCHITECTURE, AND THE THREE THINGS THAT SURPRISE PEOPLE
------------------------------------------------------
The expert runs in LOCKSTEP with the frozen VLM: expert layer i reads the keys
and values the VLM produced at ITS layer i, not the VLM's final output. Layers
alternate:

    layer i, i % self_attn_every_n_layers == 0  ->  FULL attention
        action queries attend over [prefix keys | action keys]. Action tokens
        see the observation AND each other.
    otherwise                                  ->  CROSS attention
        action queries attend over prefix keys only. No action-to-action path
        in this layer at all.

1. The "self-attention" layers are really FULL attention (prefix + actions),
   not action-only self-attention. Pure cross-attention layers give the action
   tokens no way to coordinate with each other; pure joint self-attention on
   the concatenated sequence would recompute the ~240-token prefix on every one
   of the 10 Euler steps. The interleave buys chunk coherence at cross-attention
   cost.
2. The action chunk is CAUSAL, not bidirectional. `att_masks` is all-ones over
   the 50 action tokens, so action token k attends to 0..k only. Inherited from
   openpi/pi0. A bidirectional chunk is the obvious alternative and is a clean
   ablation this file supports via `causal_chunk=False`.
3. Cross-attention layers reset the action tokens' RoPE positions to 0..49,
   while full-attention layers give them prefix_len..prefix_len+49. The same
   token therefore carries two different position conventions depending on the
   layer type. It is consistent between training and inference, so the weights
   are self-consistent, but it is not something anyone would design on purpose.
   `reset_cross_positions=False` turns it off.

At cross layers the expert does NOT get its own keys from the prefix hidden
states: it takes the VLM's already-computed per-layer K/V (5 heads x 64) and
runs them through its own 320->320 adapter. That is why the prefix cache is
worth keeping per layer and why `FrozenSmolVLM.encode_prefix` returns K/V
rather than hidden states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from config import SmolVLAConfig
from vlm_backbone import FrozenSmolVLM, PrefixCache, apply_rope, make_att_2d_masks


def sinusoidal_time_embedding(
    t: Tensor, dim: int, min_period: float, max_period: float
) -> Tensor:
    """Log-spaced sin/cos embedding of a scalar in [0, 1].

    Not the transformer's integer-position table: the flow time is continuous,
    and with 10 inference steps the network must distinguish t=0.9 from t=0.8,
    so the shortest period (4e-3) sits well below the step size while the
    longest (4.0) spans the whole interval. Getting this band wrong is a quiet
    failure mode -- the network simply stops being able to tell early steps
    from late ones and regresses towards the mean velocity.
    """
    if dim % 2:
        raise ValueError(f"dim must be even, got {dim}")
    # float64 for the period ladder, matching lerobot: min_period=4e-3 to
    # max_period=4.0 spans three orders of magnitude, and 2*pi/period at the short
    # end is ~1571, so float32 rounding of the ladder itself shows up as visibly
    # uneven frequencies rather than as harmless noise.
    #
    # MPS cannot hold float64 AT ALL, so build the ladder on the CPU and move the
    # float32 result across. lerobot instead degrades the whole computation to
    # float32 on MPS (`get_safe_dtype`); doing it this way keeps Apple silicon
    # bit-identical to the CUDA path, which is the one the real runs use, so a
    # local smoke test cannot silently disagree with the GPU.
    ladder_dev = torch.device("cpu") if t.device.type == "mps" else t.device
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=ladder_dev)
    period = min_period * (max_period / min_period) ** fraction
    scale = 1.0 / period * 2 * math.pi
    # Move THEN cast: a single .to(device=..., dtype=...) casts on the source
    # device first, which is exactly the thing MPS cannot do.
    x = scale[None, :] * t[:, None].to(ladder_dev).to(torch.float64)
    emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1).to(torch.float32)
    return emb.to(t.device)


def get_intermediate_size(hidden: int, multiplier: int = 4, multiple_of: int = 256) -> int:
    """Llama-style SwiGLU FFN width: 2/3 * 4h, rounded up to a multiple of 256.

    The 2/3 is there because SwiGLU uses three matrices (gate, up, down) where a
    plain FFN uses two, so shrinking the inner width by 2/3 keeps the parameter
    count comparable to a 4h ReLU MLP. For h=720 this gives 2048.
    """
    hidden = int(2 * hidden / 3)
    hidden = int(multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class RMSNorm(nn.Module):
    """Llama's normalisation: rescale by RMS, no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class ExpertGeometry:
    """Everything the expert needs to know about the frozen backbone."""
    d_vlm: int
    d_expert: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    ffn_hidden: int
    rms_eps: float
    rope_theta: float

    @classmethod
    def from_backbone(cls, vlm: FrozenSmolVLM, cfg: SmolVLAConfig) -> ExpertGeometry:
        d_expert = int(vlm.hidden_size * cfg.expert_width_multiplier)
        n_layers = vlm.num_layers if cfg.num_expert_layers <= 0 else cfg.num_expert_layers
        if vlm.num_layers % n_layers:
            raise ValueError(
                f"num_expert_layers={n_layers} must divide num_vlm_layers={vlm.num_layers}"
            )
        return cls(
            d_vlm=vlm.hidden_size,
            d_expert=d_expert,
            num_heads=vlm.num_heads,
            num_kv_heads=vlm.num_kv_heads,
            head_dim=vlm.head_dim,
            num_layers=n_layers,
            ffn_hidden=get_intermediate_size(d_expert),
            rms_eps=float(getattr(vlm.vlm.config.text_config, "rms_norm_eps", 1e-5)),
            rope_theta=vlm.rope_theta,
        )

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def q_width(self) -> int:
        return self.num_heads * self.head_dim


class ExpertAttention(nn.Module):
    """One attention block of the expert, either FULL or CROSS.

    Note the asymmetric widths. Queries are projected to the VLM's full
    multi-head width (15 x 64 = 960) even though the expert's residual stream is
    only 720 wide, because the keys it must dot against live in the frozen VLM's
    64-dim head space. Grouped-query attention (15 query heads sharing 5 KV
    heads) is inherited from the backbone for the same reason.
    """

    def __init__(self, geo: ExpertGeometry, mode: str):
        super().__init__()
        if mode not in ("full", "cross"):
            raise ValueError(mode)
        self.mode = mode
        self.geo = geo
        self.q_proj = nn.Linear(geo.d_expert, geo.q_width, bias=False)
        # A FULL layer builds keys from its own action tokens (720 -> 320).
        # A CROSS layer instead adapts the frozen VLM's keys (320 -> 320): the
        # prefix K/V are already computed and cached, so the expert only has to
        # learn a projection into its own attention space.
        kv_in = geo.d_expert if mode == "full" else geo.kv_width
        self.k_proj = nn.Linear(kv_in, geo.kv_width, bias=False)
        self.v_proj = nn.Linear(kv_in, geo.kv_width, bias=False)
        self.o_proj = nn.Linear(geo.q_width, geo.d_expert, bias=False)

    def forward(
        self,
        x: Tensor,                  # (B, S, d_expert) action tokens
        prefix: PrefixCache,
        layer_idx: int,
        action_positions: Tensor,   # (B, S)
        att_mask: Tensor,           # (B, 1, S, S + L) for full, (B, 1, S, L) for cross
    ) -> Tensor:
        geo = self.geo
        b, s = x.shape[:2]
        q = self.q_proj(x).view(b, s, geo.num_heads, geo.head_dim)
        q = apply_rope(q, action_positions, geo.rope_theta)

        pk = prefix.keys[layer_idx]                       # (B, L, kv, hd), RoPE already applied
        pv = prefix.values[layer_idx]

        if self.mode == "full":
            # Prefix K/V are already in head space, so they bypass k_proj/v_proj
            # entirely (which here map FROM the 720-wide expert stream). Only the
            # action tokens are projected, and only they need RoPE applied now --
            # the prefix keys were rotated when the cache was built.
            k_act = self.k_proj(x).view(b, s, geo.num_kv_heads, geo.head_dim)
            v_act = self.v_proj(x).view(b, s, geo.num_kv_heads, geo.head_dim)
            k_act = apply_rope(k_act, action_positions, geo.rope_theta)
            k = torch.cat([pk, k_act], dim=1)
            v = torch.cat([pv, v_act], dim=1)
        else:
            # Cross layer: k_proj/v_proj are 320 -> 320 adapters on the frozen
            # VLM's own keys and values. No action-to-action path in this layer.
            k = self.k_proj(pk.reshape(*pk.shape[:2], -1)).view(b, -1, geo.num_kv_heads, geo.head_dim)
            v = self.v_proj(pv.reshape(*pv.shape[:2], -1)).view(b, -1, geo.num_kv_heads, geo.head_dim)

        groups = geo.num_heads // geo.num_kv_heads
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.repeat_interleave(groups, dim=2).transpose(1, 2),
            v.repeat_interleave(groups, dim=2).transpose(1, 2),
            attn_mask=att_mask,
        )
        att = att.transpose(1, 2).reshape(b, s, geo.q_width)
        return self.o_proj(att)


class ExpertLayer(nn.Module):
    def __init__(self, geo: ExpertGeometry, mode: str):
        super().__init__()
        self.input_layernorm = RMSNorm(geo.d_expert, geo.rms_eps)
        self.self_attn = ExpertAttention(geo, mode)
        self.post_attention_layernorm = RMSNorm(geo.d_expert, geo.rms_eps)
        self.mlp = SwiGLU(geo.d_expert, geo.ffn_hidden)

    @property
    def mode(self) -> str:
        return self.self_attn.mode

    def forward(self, x, prefix, layer_idx, action_positions, att_mask):
        x = x + self.self_attn(
            self.input_layernorm(x), prefix, layer_idx, action_positions, att_mask
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class ActionExpert(nn.Module):
    """The trained half: a narrow transformer over the 50 action tokens."""

    def __init__(self, geo: ExpertGeometry, self_attn_every_n_layers: int):
        super().__init__()
        self.geo = geo
        modes = [
            "full" if (self_attn_every_n_layers > 0 and i % self_attn_every_n_layers == 0) else "cross"
            for i in range(geo.num_layers)
        ]
        self.layers = nn.ModuleList(ExpertLayer(geo, m) for m in modes)
        self.norm = RMSNorm(geo.d_expert, geo.rms_eps)
        self.modes = modes

    def forward(
        self,
        x: Tensor,
        prefix: PrefixCache,
        full_mask: Tensor,
        cross_mask: Tensor,
        full_positions: Tensor,
        cross_positions: Tensor,
    ) -> Tensor:
        # The expert may be shallower than the VLM; then each expert layer reads
        # the VLM layer `i * stride`, so a 4-layer expert on a 16-layer VLM taps
        # depths 0, 4, 8, 12 rather than crowding into the bottom four.
        stride = max(1, len(prefix.keys) // self.geo.num_layers)
        for i, layer in enumerate(self.layers):
            vlm_layer = i * stride
            if layer.mode == "full":
                x = layer(x, prefix, vlm_layer, full_positions, full_mask)
            else:
                x = layer(x, prefix, vlm_layer, cross_positions, cross_mask)
        return self.norm(x)


class SmolVLAFromScratch(nn.Module):
    """Frozen SmolVLM2 + hand-written action expert + flow matching."""

    def __init__(self, cfg: SmolVLAConfig, vlm: FrozenSmolVLM | None = None):
        super().__init__()
        self.cfg = cfg
        self.vlm = vlm if vlm is not None else FrozenSmolVLM(
            cfg.vlm_model_name,
            num_layers=cfg.num_vlm_layers,
            rope_theta=cfg.rope_theta,
        )
        self.geo = ExpertGeometry.from_backbone(self.vlm, cfg)
        g = self.geo

        # The state token joins the PREFIX, so it is projected to the VLM width,
        # not the expert width. It is the one prefix component that is trained:
        # a frozen VLM has no idea what a 7-DoF joint vector means, and 32 -> 960
        # is cheap.
        self.state_proj = nn.Linear(cfg.max_state_dim, g.d_vlm)
        self.action_in_proj = nn.Linear(cfg.max_action_dim, g.d_expert)
        # Time is fused with the action embedding by concatenation + a 2-layer
        # MLP rather than added like a positional code. Adding would force the
        # time signal to share the action subspace; concatenation lets the MLP
        # learn a genuine interaction, which matters because the velocity field
        # depends jointly on where you are and when you are.
        self.action_time_mlp_in = nn.Linear(2 * g.d_expert, g.d_expert)
        self.action_time_mlp_out = nn.Linear(g.d_expert, g.d_expert)
        self.expert = ActionExpert(g, cfg.self_attn_every_n_layers)
        self.action_out_proj = nn.Linear(g.d_expert, cfg.max_action_dim)

        # Behavioural switches for the ablations described in the module docstring.
        self.causal_chunk = True
        self.reset_cross_positions = True

    # ------------------------------------------------------------------ prefix
    def embed_prefix(
        self,
        images: list[Tensor],       # each (B, 3, 512, 512) in [-1, 1]
        img_masks: list[Tensor],    # each (B,) bool
        lang_tokens: Tensor,        # (B, T) int64
        lang_masks: Tensor,         # (B, T) bool
        state: Tensor,              # (B, max_state_dim)
    ) -> tuple[Tensor, Tensor, Tensor]:
        embs, pads, att = [], [], []
        for img, mask in zip(images, img_masks, strict=True):
            e = self.vlm.embed_images(img)
            embs.append(e)
            pads.append(mask[:, None].expand(e.shape[0], e.shape[1]))
            att += [0] * e.shape[1]

        lang = self.vlm.embed_text(lang_tokens)
        embs.append(lang)
        pads.append(lang_masks)
        att += [0] * lang.shape[1]

        st = self.state_proj(state)
        st = st[:, None, :] if st.ndim == 2 else st
        embs.append(st)
        pads.append(torch.ones(st.shape[:2], dtype=torch.bool, device=st.device))
        # att=1 opens a new block at the state token, so images and language
        # cannot attend to it. Not for causality's sake -- it keeps the visual
        # and linguistic representation independent of the proprioceptive one,
        # which is what makes the image/language part of the prefix reusable.
        att += [1] * st.shape[1]

        embs = torch.cat([e.to(embs[-1].dtype) for e in embs], dim=1)
        pads = torch.cat(pads, dim=1)
        att = torch.tensor(att, dtype=torch.int64, device=pads.device)[None].expand(
            pads.shape[0], -1
        )
        return embs, pads, att

    def encode_prefix(self, images, img_masks, lang_tokens, lang_masks, state) -> PrefixCache:
        """Build and encode the prefix.

        The gradient subtlety: every VLM parameter has requires_grad=False, but
        `state_proj` is trainable and its output is a prefix token, so the 16
        frozen layers still have to keep activations for the backward pass just
        to deliver a gradient to a 32x960 matrix. That is the price lerobot pays
        too. With `train_state_proj=False` the whole prefix runs under no_grad
        and the memory cost disappears -- worth knowing when an L4 runs out of
        room, at the cost of the model never learning what a joint angle means.
        """
        embs, pads, att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        if self.cfg.train_state_proj and torch.is_grad_enabled():
            return self.vlm.encode_prefix(embs, pads, att)
        with torch.no_grad():
            return self.vlm.encode_prefix(embs, pads, att)

    # ------------------------------------------------------------------ suffix
    def embed_suffix(self, x_t: Tensor, t: Tensor) -> Tensor:
        a = self.action_in_proj(x_t)
        te = sinusoidal_time_embedding(
            t, self.geo.d_expert, self.cfg.min_period, self.cfg.max_period
        ).to(a.dtype)
        h = torch.cat([a, te[:, None, :].expand_as(a)], dim=2)
        return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(h)))

    def _masks_and_positions(self, prefix: PrefixCache, chunk: int, device):
        b, prefix_len = prefix.pad_mask.shape
        suffix_att = torch.ones(b, chunk, dtype=torch.int64, device=device)
        if not self.causal_chunk:
            suffix_att = torch.zeros_like(suffix_att)
        suffix_pad = torch.ones(b, chunk, dtype=torch.bool, device=device)
        # rows = action tokens; every real prefix token is visible to all of them
        prefix_block = prefix.pad_mask[:, None, :].expand(b, chunk, prefix_len)
        suffix_block = make_att_2d_masks(suffix_pad, suffix_att)
        full_mask = torch.cat([prefix_block, suffix_block], dim=2)[:, None]
        cross_mask = prefix_block[:, None]

        offsets = prefix.pad_mask.sum(-1, keepdim=True)
        full_positions = offsets + torch.arange(chunk, device=device)[None, :]
        cross_positions = (
            torch.arange(chunk, device=device)[None, :].expand(b, chunk)
            if self.reset_cross_positions
            else full_positions
        )
        return full_mask, cross_mask, full_positions, cross_positions

    def velocity(self, prefix: PrefixCache, x_t: Tensor, t: Tensor) -> Tensor:
        """v(x_t, t | observation): the field we train and then integrate."""
        h = self.embed_suffix(x_t, t)
        full_mask, cross_mask, full_pos, cross_pos = self._masks_and_positions(
            prefix, x_t.shape[1], x_t.device
        )
        out = self.expert(h, prefix, full_mask, cross_mask, full_pos, cross_pos)
        return self.action_out_proj(out.float())

    # ------------------------------------------------------------------- train
    def sample_time(self, batch: int, device) -> Tensor:
        """Beta(1.5, 1.0) on [0.001, 1.0]: oversample the NOISY end.

        Under this file's convention t=1 is pure noise. A right-skewed Beta puts
        more samples there, which is where the field has to make its hardest
        decision -- which mode of the action distribution to commit to. Late
        (near-data) steps are comparatively easy local corrections. Uniform t
        trains a worse model for the same compute; that is one of the few things
        the flow-matching literature agrees on.
        """
        dist = torch.distributions.Beta(
            concentration1=self.cfg.time_beta_alpha, concentration0=self.cfg.time_beta_beta
        )
        t = dist.sample((batch,)).to(device=device, dtype=torch.float32)
        return t * self.cfg.time_scale + self.cfg.time_shift

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions: Tensor,             # (B, chunk, max_action_dim), normalised
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        t: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        if noise is None:
            noise = torch.randn(actions.shape, dtype=torch.float32, device=actions.device)
        if t is None:
            t = self.sample_time(actions.shape[0], actions.device)

        te = t[:, None, None]
        x_t = te * noise + (1.0 - te) * actions
        # The regression target. Constant in t along a conditional path -- which
        # is the entire trick: a network that sees only (x_t, t) cannot know
        # which (noise, action) pair produced x_t, so regressing this per-sample
        # target makes it learn the CONDITIONAL EXPECTATION over all pairs, and
        # that expectation is exactly the marginal velocity field that transports
        # noise to data. See FLOW_MATCHING.md section 3.
        u_t = noise - actions

        v_t = self.velocity(
            self.encode_prefix(images, img_masks, lang_tokens, lang_masks, state), x_t, t
        )
        losses = F.mse_loss(u_t, v_t, reduction="none")

        # Two masks, for two different kinds of fake data:
        #   dims  : we padded 8 real dims out to 32; the other 24 carry no signal
        #   frames: chunks that ran past the end of an episode were edge-padded
        losses = losses[..., : self.cfg.action_dim]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad)[:, :, None]
            denom = (~actions_is_pad).sum().clamp(min=1) * self.cfg.action_dim
            loss = losses.sum() / denom
        else:
            loss = losses.mean()
        return loss, {"loss": loss.item(), "t_mean": t.mean().item()}

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise: Tensor | None = None
    ) -> Tensor:
        """Euler-integrate the velocity field from noise (t=1) to actions (t=0)."""
        b = state.shape[0]
        # ONE VLM forward for the whole chunk, reused across every Euler step.
        prefix = self.encode_prefix(images, img_masks, lang_tokens, lang_masks, state)
        if noise is None:
            noise = torch.randn(
                (b, self.cfg.chunk_size, self.cfg.max_action_dim),
                dtype=torch.float32,
                device=state.device,
            )
        n = self.cfg.num_inference_steps
        dt = -1.0 / n
        x_t = noise
        for step in range(n):
            t = torch.full((b,), 1.0 + step * dt, dtype=torch.float32, device=state.device)
            x_t = x_t + dt * self.velocity(prefix, x_t, t)
        return x_t

    # ------------------------------------------------------------------ utils
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        modes = "".join("F" if m == "full" else "C" for m in self.expert.modes)
        return (
            f"expert layers [{modes}] (F=full, C=cross), d_expert={self.geo.d_expert}, "
            f"ffn={self.geo.ffn_hidden}\n"
            f"trainable {trainable / 1e6:.1f}M | frozen {frozen / 1e6:.1f}M | "
            f"total {(trainable + frozen) / 1e6:.1f}M"
        )
