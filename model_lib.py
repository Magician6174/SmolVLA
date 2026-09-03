"""SmolVLA the other way round: lerobot's implementation, lerobot's pretrained weights.

`model.py` writes the action expert, the flow-matching loss and the Euler
integrator by hand. This file writes none of that. It wraps lerobot's
`VLAFlowMatching` and optionally initialises it from `lerobot/smolvla_base` --
the 450M checkpoint pretrained for 200k steps on ~23k community robot episodes.

Both classes expose the same three methods (`forward`, `sample_actions`,
`param_summary`) taking the same arguments, so `train.py --build {scratch,
finetune}` and `rollout.py` are shared verbatim. That is the whole point: any
difference in the final success rate is attributable to the model, not to
different data plumbing, normalisation, chunking or rollout policy.

WHAT THE THREE BUILDS SEPARATE
------------------------------
    scratch                     my expert,      random init
    finetune --pretrained none  lerobot expert, random init
    finetune (default)          lerobot expert, smolvla_base weights

The middle one exists because without it a gap between the first and third
confounds two very different claims: "I implemented the architecture correctly"
and "large-scale robot pretraining transfers to my task". Running all three
separates them. If scratch ~= pretrained-none, my implementation is faithful; if
pretrained >> both, the 23k episodes are doing the work. On 184 demonstrations of
a single task I expect the pretrained build to win clearly, and being able to say
*which* factor won is worth one extra training run.

WHY NOT `SmolVLAPolicy.from_pretrained`
---------------------------------------
That is the documented entry point, but it drags in `PreTrainedPolicy`, the
dataset-feature validation and lerobot's own processor pipeline (normalisation,
tokenisation, device placement) -- all of which `dataset.py` already does, and
which we need to keep identical across both builds. So we construct the inner
`VLAFlowMatching` directly and load the checkpoint's tensors ourselves. That also
makes the load explicit enough to assert on, which caught the geometry
differences documented in `_assert_compatible`.

ONE REAL INCOMPATIBILITY, HANDLED
---------------------------------
`smolvla_base` was pretrained on 6-DoF arms with 3 cameras at 256x256 and
`pad_language_to="max_length"`. None of that constrains us: state and action are
zero-padded to 32 dims either way (so a 6-DoF checkpoint's `state_proj` accepts
our 8 dims), the vision tower is resolution-agnostic after letterboxing to
512x512, and "longest" vs "max_length" language padding is numerically identical
because padded tokens are masked as keys and excluded from the RoPE position
count. Verified in `parity_test.py` rather than asserted here.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from config import SmolVLAConfig

PRETRAINED_REPO = "lerobot/smolvla_base"


def _lerobot_config(cfg: SmolVLAConfig):
    """Translate our config into lerobot's, then check nothing was lost.

    Only the fields `VLAFlowMatching.__init__` actually reads are set. lerobot's
    config also carries dataset features, optimiser presets and an RTC block; we
    supply none of them because we own the optimiser (train.py) and the data
    (dataset.py), and RTC (real-time chunking) is a deployment feature for
    physical robots with control-loop latency, which a simulator does not have.
    """
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as LeRobotConfig

    lr_cfg = LeRobotConfig(
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        resize_imgs_with_padding=tuple(cfg.resize_imgs_with_padding),
        tokenizer_max_length=cfg.tokenizer_max_length,
        pad_language_to=cfg.pad_language_to,
        num_steps=cfg.num_inference_steps,
        vlm_model_name=cfg.vlm_model_name,
        num_vlm_layers=cfg.num_vlm_layers,
        num_expert_layers=cfg.num_expert_layers,
        self_attn_every_n_layers=cfg.self_attn_every_n_layers,
        expert_width_multiplier=cfg.expert_width_multiplier,
        attention_mode=cfg.attention_mode,
        add_image_special_tokens=cfg.add_image_special_tokens,
        freeze_vision_encoder=cfg.freeze_vlm,
        train_expert_only=cfg.freeze_vlm,
        train_state_proj=cfg.train_state_proj,
        load_vlm_weights=True,       # the frozen backbone must be pretrained in every build
        min_period=cfg.min_period,
        max_period=cfg.max_period,
        device=cfg.device,
    )
    return lr_cfg


def _assert_compatible(cfg: SmolVLAConfig) -> None:
    """Geometry that `smolvla_base`'s weights physically require.

    These are shape constraints, not preferences: the checkpoint's `state_proj`
    is 32x960 and its expert layers are 720 wide. Changing any of them turns the
    weight load into a silent partial load, which is the failure mode this
    function exists to convert into a stack trace.
    """
    required = {
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_vlm_layers": 16,
        "expert_width_multiplier": 0.75,
        "self_attn_every_n_layers": 2,
        "chunk_size": 50,
    }
    for key, want in required.items():
        got = getattr(cfg, key)
        if got != want:
            raise ValueError(
                f"{PRETRAINED_REPO} was pretrained with {key}={want}; config says {got}. "
                f"Its weights cannot be loaded at that geometry. Use --build scratch, "
                f"or --pretrained none to train lerobot's architecture from random init."
            )


def _pretrained_state_dict(repo: str = PRETRAINED_REPO) -> dict[str, Tensor]:
    """Load `model.safetensors` and strip the `model.` prefix that
    `SmolVLAPolicy` adds (its `VLAFlowMatching` lives at `self.model`)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Resolves straight out of the HF cache under HF_HUB_OFFLINE=1, so training
    # nodes without egress still work once the checkpoint has been synced.
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    return {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}


class SmolVLAFinetune(nn.Module):
    """lerobot's `VLAFlowMatching`, wrapped to match `SmolVLAFromScratch`'s API."""

    def __init__(self, cfg: SmolVLAConfig, pretrained: str | None = PRETRAINED_REPO):
        super().__init__()
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

        self.cfg = cfg
        if pretrained:
            _assert_compatible(cfg)
        self.flow = VLAFlowMatching(_lerobot_config(cfg))
        self.pretrained = pretrained

        if pretrained:
            sd = _pretrained_state_dict(pretrained)
            missing, unexpected = self.flow.load_state_dict(sd, strict=False)
            # The truncated VLM means the checkpoint has no tensors for layers
            # 16..31, and lerobot drops lm_head; anything else missing is a bug.
            real_missing = [k for k in missing if "lm_head" not in k]
            if real_missing or unexpected:
                raise RuntimeError(
                    f"{pretrained} load mismatch:\n  missing={real_missing[:8]}\n"
                    f"  unexpected={unexpected[:8]}"
                )
            self.loaded_tensors = len(sd)
        else:
            self.loaded_tensors = 0

        # ------------------------------------------------------------------
        # FORCE float32. This is load-bearing, not tidiness.
        #
        # lerobot loads the VLM with `torch_dtype="bfloat16"`
        # (smolvlm_with_expert.py:92) and the action expert, being built from a
        # config derived from it, inherits bf16. The result is that 129 of the
        # 155 TRAINABLE tensors are bf16 MASTER weights, while the 26 tensors
        # lerobot constructs itself (state_proj, the action projections, the
        # time MLP) stay fp32.
        #
        # bf16 has 8 mantissa bits, so it resolves ~3 decimal digits. At lr=1e-4
        # on a weight of order 1e-2 the AdamW update is ~1e-6 RELATIVE, which
        # rounds straight to zero: most of the expert would simply not train,
        # while the loss curve still looks plausible because the fp32 minority
        # keeps moving. Measured directly: the finetune checkpoint was 206 MB
        # against the from-scratch build's 400 MB for an identical 99,880,992
        # parameters.
        #
        # It is also a confound even where it does train, because the
        # from-scratch build keeps its frozen backbone in fp32. Casting the
        # whole module makes the two builds dtype-identical, which is the entire
        # point of running them side by side. Speed is unaffected: bf16 compute
        # still happens inside torch.autocast on CUDA. The cost is ~1.4 GB of
        # frozen weights instead of ~0.7 GB, which fits an L4 comfortably.
        self.flow.to(torch.float32)

    # --------------------------------------------------------------- training
    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions: Tensor,
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        t: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Same loss reduction as `model.py`, applied to lerobot's per-element MSE.

        `VLAFlowMatching.forward` returns unreduced losses of shape
        (B, chunk, 32). We deliberately do NOT use `SmolVLAPolicy.forward`'s
        reduction: it slices to the real action dim, multiplies by the pad mask,
        then re-slices to `max_action_dim` (a no-op) and divides by a denominator
        computed from `losses.shape[-1]`, which is the real dim by then. The
        arithmetic happens to work out, but reimplementing it here guarantees the
        two builds are compared under a bit-identical scalar, not two loss
        functions that agree by coincidence.
        """
        if t is None:
            t = self.flow.sample_time(actions.shape[0], actions.device)
        losses = self.flow.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, t
        )
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
        return self.flow.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

    # ------------------------------------------------------------------ utils
    def param_summary(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        expert = self.flow.vlm_with_expert.expert_hidden_size
        src = self.pretrained or "random init"
        return (
            f"lerobot VLAFlowMatching from {src} ({self.loaded_tensors} tensors), "
            f"d_expert={expert}\n"
            f"trainable {trainable / 1e6:.1f}M | frozen {frozen / 1e6:.1f}M | "
            f"total {(trainable + frozen) / 1e6:.1f}M"
        )
