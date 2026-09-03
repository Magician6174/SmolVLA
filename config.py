"""SmolVLA configuration for the Panda pick-and-place task.

Single source of truth for dims, architecture, and training hyperparameters.
Defaults follow the SmolVLA paper (Shukor et al., 2506.01844) and lerobot's
`SmolVLAConfig`, adapted to our 8-DoF / 3-camera setup. Where the paper and
lerobot disagree, the discrepancy is called out inline and FLOW_MATCHING.md
section 7.1 has the full accounting.

Mental model (how this differs from our other two policies):
  ACT       : CVAE. One observation -> a 100-step action chunk in ONE forward
              pass; temporally ensemble overlapping chunks at test time.
  Diffusion : DDPM. Start from pure noise and iteratively DENOISE a 16-step
              action trajectory over 100 (train) / 10 (DDIM) reverse steps.
              The net predicts the NOISE that was added.
  SmolVLA   : Flow matching. Also starts from noise, but the net predicts a
              VELOCITY field v(x_t, t) and we integrate it with plain Euler.
              The path from noise to data is a STRAIGHT LINE by construction
              (x_t = t*noise + (1-t)*action), which is why 10 Euler steps
              suffice where DDPM needed a learned variance schedule.
              Conditioning is not a small ResNet: it is a frozen 16-layer
              vision-language model, and the language token stream is a real
              input, not decoration.

THREE THINGS THAT ARE EASY TO GET BACKWARDS
-------------------------------------------
1. TIME DIRECTION. lerobot's code (and therefore this project) uses
   `x_t = t*noise + (1-t)*action`, so **t=1 is pure noise and t=0 is data**,
   and inference integrates DOWNWARD with `dt = -1/num_steps`. The paper's
   section 3.1 writes the interpolation the same way but gives the target as
   `eps - A` while calling it `A - eps` in prose; the code is self-consistent,
   so we follow the code.
2. STATE/ACTION PADDING. Vectors are zero-padded to 32 dims. Our robot needs
   only 8. This is not laziness: `smolvla_base` was pretrained with 32-wide
   input/output projections, so the finetuning build cannot load those weights
   at any other width. The from-scratch build keeps 32 too, purely so the two
   models are numerically comparable rather than differing in two ways at once.
   The loss is masked to the real 8 dims either way (see model.py).
3. n_action_steps. It affects ROLLOUT ONLY, never training. The paper uses the
   full 50-step chunk open-loop on real robots but **n_action_steps=1 in
   simulation** (re-observe every frame). lerobot's default is 50. We default
   to 1 (the paper's sim setting, and the regime our ACT/DP numbers were
   measured in) and report both.
"""
from dataclasses import asdict, dataclass, field

import torch


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SmolVLAConfig:
    # --- data -------------------------------------------------------------
    repo_id: str = "panda_pick_place"
    # Local path the dataset is synced to (from S3) before training. Random-access
    # video decode over S3 is far too slow, so we always read from local disk.
    data_root: str = "data/panda_pick_place"
    cameras: tuple = ("front", "diag", "wrist")
    state_dim: int = 8                 # 7 arm joints + gripper finger pos (REAL dims)
    action_dim: int = 8                # 7 arm joint targets + gripper ctrl (REAL dims)
    # Padded width the model actually sees. See note 2 in the module docstring.
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_hw: tuple = (480, 640)       # native recorded resolution (H, W)
    fps: int = 30

    # SmolVLM2's vision tower is trained at 512x512. We letterbox (resize keeping
    # aspect ratio, then pad) rather than stretch, because our 4:3 frames would
    # otherwise be distorted relative to everything the backbone has ever seen.
    resize_imgs_with_padding: tuple = (512, 512)
    # SigLIP patch 16 -> 32x32 = 1024 patches, then SmolVLM's pixel-shuffle
    # (scale_factor=4) folds 4x4 neighbourhoods into one token -> 64 tokens per
    # camera. 3 cameras = 192 image tokens. Worth knowing before wondering why
    # a "500M" model is slow: the prefix is ~240 tokens long.

    # --- language ----------------------------------------------------------
    # The on-disk dataset has total_tasks=1: one constant instruction for all
    # 45,030 frames. A VLA trained on that is a vision-and-state policy wearing
    # a language costume. relabel_tasks.py recovers the per-episode object shape
    # from the recorder sidecar and synthesises 3 instructions. See that module
    # for the index-alignment trap and the visual gate that guards it.
    use_relabelled_instructions: bool = True
    tokenizer_max_length: int = 48      # paper/lerobot value; our strings are ~10 tokens
    pad_language_to: str = "longest"    # "longest" | "max_length"
    # Language ablation mode, applied at EVALUATION time on a fixed checkpoint:
    #   "correct"  A: the true instruction for the episode's object
    #   "mismatch" B: an instruction naming one of the other two shapes
    #   "empty"    C: the empty string
    #   "generic"  D: the original single dataset string (no shape information)
    # A > B ~ C means the policy is genuinely language-conditioned. A ~ B ~ C
    # means vision alone disambiguates and the language head is decorative.
    # A ~ B > C means language acts only as a "task is active" gate.
    language_mode: str = "correct"

    # --- VLM backbone ------------------------------------------------------
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    # Use only the FIRST N of the VLM's 32 decoder layers. The paper frames this
    # as free speed; its own Table 8 says otherwise (N=8 -> 75.0, N=16 -> 78.5,
    # N=24 -> 79.5, N=32 -> 80.3 avg success). N=16 costs ~1.8 points to halve
    # the backbone. We keep 16 because that is the geometry `smolvla_base`
    # shipped with, and the finetuning baseline must be able to load it.
    num_vlm_layers: int = 16
    freeze_vlm: bool = True             # the paper trains the expert only
    train_state_proj: bool = True       # ...except the state projection
    add_image_special_tokens: bool = False
    # RoPE base for the frozen text tower. SmolVLM2-500M was pretrained with
    # 100_000; lerobot hardcodes 10_000 and therefore so did `smolvla_base`
    # during its 200k-step pretraining. We match lerobot so the from-scratch and
    # finetuned runs share identical conditioning; "checkpoint" is the ablation
    # that feeds the VLM the frequencies it was actually trained on.
    rope_theta: float | str = 10_000.0

    # --- action expert -----------------------------------------------------
    # A narrow transformer that cross-attends into the frozen VLM's KV cache.
    # Width = int(vlm_hidden * multiplier) = int(960 * 0.75) = 720.
    # Table 9 of the paper: x1.00 -> 82.3, x0.75 -> 77.5, x0.50 -> 80.3,
    # x0.25 -> 73.8. So 0.75 is *worse* than 0.50 and is still the shipped
    # default; the paper prose, its table, SmolVLAConfig (0.75) and
    # SmolVLMWithExpertModel.__init__ (0.5) disagree three ways. We take 0.75
    # for checkpoint compatibility and flag it as a thing to sweep, not a
    # tuned optimum.
    expert_width_multiplier: float = 0.75
    num_expert_layers: int = -1         # <=0 -> same depth as the truncated VLM
    # Layer type interleave: layer_idx % self_attn_every_n_layers == 0 gets
    # SELF attention over the action tokens, every other layer CROSS-attends
    # into the VLM prefix. Pure cross-attention cannot let action tokens talk
    # to each other; pure self-attention on a concatenated sequence would make
    # the frozen prefix pay quadratic cost every denoise step. The interleave
    # buys both, and it is why the prefix KV cache can be computed ONCE per
    # observation and reused across all 10 Euler steps.
    self_attn_every_n_layers: int = 2
    attention_mode: str = "cross_attn"

    # --- chunking ----------------------------------------------------------
    n_obs_steps: int = 1                # SmolVLA conditions on the current frame only
    chunk_size: int = 50                # actions predicted per forward pass (~1.7s @30fps)

    # --- flow matching -----------------------------------------------------
    # Timestep sampling. Uniform t would spend half its samples in the
    # near-noise regime where the velocity target is almost pure noise and the
    # gradient is uninformative. Beta(1.5, 1.0) is right-skewed, so it
    # oversamples t near 1... which under the CODE's convention is the NOISY
    # end. That is deliberate: the first Euler steps out of noise are the ones
    # that decide which mode of the action distribution you land in, so they
    # get the most training signal. Rescaled to [0.001, 1.0] to avoid t=0
    # exactly (where the velocity target degenerates).
    time_beta_alpha: float = 1.5        # torch.distributions.Beta concentration1
    time_beta_beta: float = 1.0         # concentration0
    time_scale: float = 0.999
    time_shift: float = 0.001
    # Sinusoidal time embedding band. min_period/max_period set the frequency
    # range; with 10 inference steps the net must resolve dt=0.1, so the
    # highest frequency has to be well above 1/0.1.
    min_period: float = 4e-3
    max_period: float = 4.0
    # Euler integration steps at inference. 10 is the paper's number and works
    # because the conditional probability path is a straight line: the only
    # error is the curvature of the *marginal* field, not a noise schedule.
    num_inference_steps: int = 10

    # --- training ----------------------------------------------------------
    # Paper: sim finetuning is 100k steps at batch size 64 (pretraining was
    # 200k @ 256). An L4 (24 GB) will not hold bs=64 with 3 cameras at 512x512,
    # so we use bs=16 x 4 accumulation to keep the same effective batch.
    batch_size: int = 16
    grad_accum_steps: int = 4
    n_steps: int = 100_000
    num_workers: int = 4
    prefetch_factor: int = 1            # shm-bound on SageMaker Studio; see DP config
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.95)   # note beta2=0.95, not the usual 0.999
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    # Cosine decay 1e-4 -> 2.5e-6. Warmup: the paper says 100 steps, lerobot's
    # config says 1000. We take lerobot's, since 100 steps of warmup at bs=64
    # is implausibly short for a fresh 100M-parameter transformer.
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 2.5e-6
    val_fraction: float = 0.1
    seed: int = 1000
    # Mixed precision. The VLM is frozen, so most of the graph is inference-only
    # and bf16 there is nearly free. Auto-disabled off CUDA.
    use_amp: bool = True
    # No EMA. Diffusion Policy needed it badly (its raw weights are noisy), but
    # flow matching with a straight-line path has a far better conditioned loss
    # and neither the paper nor lerobot uses EMA for SmolVLA. Not carrying it
    # over is a deliberate choice, not an omission.

    # --- rollout -----------------------------------------------------------
    # See note 3 in the module docstring: 1 = the paper's simulation setting
    # (re-observe every frame), 50 = fully open-loop chunk (lerobot's default).
    n_action_steps: int = 1
    rollout_episodes: int = 200
    rollout_max_steps: int = 500

    # --- logging / io ------------------------------------------------------
    device: str = field(default_factory=_auto_device)
    output_dir: str = "checkpoints"
    log_freq: int = 100
    val_batches: int = 25       # fixed count so val loss is comparable across runs
    val_freq: int = 1000
    save_freq: int = 5000

    def __post_init__(self):
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size "
                f"({self.chunk_size}): you cannot execute more actions than were predicted."
            )
        if self.state_dim > self.max_state_dim or self.action_dim > self.max_action_dim:
            raise ValueError(
                f"real dims ({self.state_dim}, {self.action_dim}) exceed the padded widths "
                f"({self.max_state_dim}, {self.max_action_dim})."
            )
        if self.language_mode not in ("correct", "mismatch", "empty", "generic"):
            raise ValueError(f"unknown language_mode {self.language_mode!r}")
        # AMP only makes sense on CUDA; MPS/CPU autocast is flaky or pointless.
        if self.device != "cuda":
            self.use_amp = False

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def to_dict(self) -> dict:
        return asdict(self)
