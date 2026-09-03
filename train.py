"""Train SmolVLA on the Panda pick-and-place dataset.

Offline imitation, flow-matching objective: sample a chunk of 50 actions, mix it
with Gaussian noise at a random time t, ask the expert for the velocity, and
regress `noise - actions` with masked MSE. The simulator is never opened here;
closed-loop success is measured by rollout.py, and that number is the only one
worth putting on a resume.

    python train.py --smoke                        # 4 steps, CPU, shapes only
    python train.py --build scratch  --n_steps 100000
    python train.py --build finetune --n_steps 100000   # lerobot smolvla_base

FIVE THINGS THAT DIFFER FROM OUR ACT / DIFFUSION TRAINERS
---------------------------------------------------------
1. GRADIENT ACCUMULATION IS LOAD-BEARING, not a nicety. The paper finetunes at
   batch size 64; three 512x512 images through a 500M VLM will not fit 64-wide
   on a 24 GB L4, so we run 16 x 4. The effective batch is what the LR schedule
   was tuned for, so this is a memory trick, not a hyperparameter change.
2. NO EMA. Diffusion Policy needed it (its raw weights are genuinely worse), and
   we carried it there. Neither the SmolVLA paper nor lerobot uses EMA, and the
   straight-line flow path gives a much better conditioned loss surface, so
   there is nothing to smooth. Leaving it out is a decision, not an oversight.
3. CHECKPOINTS STORE ONLY TRAINABLE TENSORS. 100M trained against 303M frozen:
   writing the whole model every 5k steps would be 1.6 GB a file for 1.6 GB of
   bytes that are already on disk in the HF cache and bit-identical every time.
   `load_policy` rebuilds the frozen backbone from the hub and layers the
   trained tensors on top. The normaliser's buffers ARE saved -- they are
   dataset statistics, not weights, and a checkpoint restored with different
   ones silently produces offset joint targets.
4. VALIDATION USES FIXED NOISE AND FIXED t. The flow-matching loss is an
   expectation over (noise, t); evaluating it with fresh random draws every time
   gives a val curve whose step-to-step variance swamps the trend, which is
   exactly the complaint written into Diffusion/train.py. Seeding a generator
   per validation batch makes the comparison across checkpoints paired: the same
   sample is always scored at the same point on the same path. The absolute
   number stops being an unbiased estimate of the true expected loss, but we
   only ever use it as a relative signal.
5. bf16 AUTOCAST, NO GradScaler. Most of this graph is a frozen forward pass, so
   the precision question is really "can the frozen VLM's activations survive
   16 bits", and bf16's exponent range means yes without loss scaling. fp16
   would need a scaler and would risk overflow in the sqrt(960)-scaled residual
   stream (see vlm_backbone.py, quirk 1).
"""
import argparse
import csv
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import torch

from config import SmolVLAConfig
from dataset import Normalizer, build_model_inputs, make_loaders, make_tokenizer


def parse_args() -> tuple[SmolVLAConfig, argparse.Namespace]:
    p = argparse.ArgumentParser()
    for f in fields(SmolVLAConfig):
        t = f.type if not isinstance(f.type, str) else eval(f.type)  # noqa: S307
        if t in (int, float, str):
            p.add_argument(f"--{f.name}", type=t)
        elif t is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
    p.add_argument(
        "--build",
        choices=("scratch", "finetune"),
        default="scratch",
        help="scratch = model.py (hand-written expert, random init); "
             "finetune = model_lib.py (lerobot smolvla_base weights)",
    )
    p.add_argument(
        "--pretrained",
        default="lerobot/smolvla_base",
        help="--build finetune only. 'none' = lerobot's architecture at random init, "
             "which isolates 'is my implementation faithful' from 'does pretraining help'",
    )
    p.add_argument("--resume", type=str, default=None, help="path to a last.pt to continue from")
    p.add_argument("--smoke", action="store_true", help="4 steps on CPU to validate the pipeline")
    args = p.parse_args()
    skip = {"smoke", "build", "resume", "pretrained"}
    cfg = SmolVLAConfig(**{k: v for k, v in vars(args).items() if k not in skip and v is not None})
    if args.smoke:
        cfg.batch_size, cfg.grad_accum_steps, cfg.num_workers = 2, 2, 0
        cfg.n_steps, cfg.val_freq, cfg.log_freq, cfg.save_freq = 4, 2, 1, 1000
        cfg.scheduler_warmup_steps = 2
        cfg.val_batches = 2     # 25 CPU val batches is 90% of a smoke run's wall time
        # The docstring promises CPU, and it has to actually be CPU: MPS autocast
        # is flaky and the point of a smoke test is to check plumbing, not speed.
        # An explicit --device still wins, so `--smoke --device mps` remains the
        # way to exercise the Apple-silicon path on purpose.
        if args.device is None:
            cfg.device, cfg.use_amp = "cpu", False
    return cfg, args


def build_policy(cfg: SmolVLAConfig, build: str, pretrained: str | None = None):
    """Both builds expose the same three methods: forward, sample_actions, param_summary."""
    if build == "scratch":
        from model import SmolVLAFromScratch

        return SmolVLAFromScratch(cfg)
    from model_lib import PRETRAINED_REPO, SmolVLAFinetune

    if pretrained is None:
        pretrained = PRETRAINED_REPO
    return SmolVLAFinetune(cfg, pretrained=None if pretrained == "none" else pretrained)


def build_optimizer(policy, cfg: SmolVLAConfig):
    """One LR group. The backbone is frozen, so there is no second group to have.

    weight_decay=1e-10 is lerobot's value and is effectively zero. It is worth
    noticing rather than copying blindly: the trained half is a fresh transformer
    where you would normally want real decay, but the *output* of that
    transformer is a velocity field with a fixed target scale, and decaying
    `action_out_proj` shrinks predicted velocities toward zero -- which biases
    the integrated trajectory toward the noise it started from.
    """
    params = [p for p in policy.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=cfg.optimizer_lr,
        betas=tuple(cfg.optimizer_betas),
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )


def build_scheduler(optimizer, cfg: SmolVLAConfig):
    """Linear warmup then cosine decay from optimizer_lr down to scheduler_decay_lr.

    The floor matters: a cosine that lands on 0 spends its last thousands of
    steps taking numerically meaningless updates. 2.5e-6 is 1/40th of peak, small
    enough to be a fine-tuning crawl and large enough to still move.
    """
    warmup, total = cfg.scheduler_warmup_steps, cfg.scheduler_decay_steps
    floor = cfg.scheduler_decay_lr / cfg.optimizer_lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def trainable_state_dict(policy) -> dict:
    """Only tensors that are trained, plus nothing that can be rebuilt. See note 3."""
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    return {k: v.cpu() for k, v in policy.state_dict().items() if k in trainable}


def save_checkpoint(path: Path, step: int, policy, normalizer, cfg, build: str, extra=None,
                    pretrained: str | None = None):
    blob = {
        "step": step,
        "build": build,
        "pretrained": pretrained,
        "model": trainable_state_dict(policy),
        "normalizer": {k: v.cpu() for k, v in normalizer.state_dict().items()},
        "config": cfg.to_dict(),
    }
    blob.update(extra or {})
    torch.save(blob, path)


def load_policy(path: str | Path, device: str | None = None, cfg_overrides: dict | None = None):
    """Rebuild (policy, normalizer, cfg) from a checkpoint. Used by rollout.py.

    `strict=False` on purpose and safely: the frozen VLM's 303M tensors are
    deliberately absent from the file, so the load reports them as missing. We
    assert that every MISSING key is frozen and that there are no UNEXPECTED
    keys, which catches a genuine mismatch while tolerating the intended one.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SmolVLAConfig(**{**blob["config"], **(cfg_overrides or {})})
    if device:
        cfg.device = device
        cfg.__post_init__()
    policy = build_policy(cfg, blob.get("build", "scratch"), blob.get("pretrained"))
    missing, unexpected = policy.load_state_dict(blob["model"], strict=False)
    frozen = {n for n, p in policy.named_parameters() if not p.requires_grad}
    buffers = set(dict(policy.named_buffers()))
    unaccounted = [k for k in missing if k not in frozen and k not in buffers]
    if unexpected or unaccounted:
        raise RuntimeError(f"checkpoint mismatch: {unexpected=} {unaccounted=}")
    normalizer = Normalizer.from_state_dict(blob["normalizer"])
    return policy.to(cfg.device).eval(), normalizer.to(cfg.device), cfg


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(policy, val_loader, cfg, tokenizer, normalizer, max_batches: int | None = None) -> float:
    """Paired validation loss: same (noise, t) for a given batch at every step."""
    policy.eval()
    # A FIXED batch count, not the whole split: combined with the fixed noise and
    # the deterministic t sweep below, this is what makes val loss comparable
    # across checkpoints instead of a resampled quantity.
    if max_batches is None:
        max_batches = cfg.val_batches
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        inputs = build_model_inputs(batch, cfg, tokenizer, normalizer, cfg.device)
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + i)
        noise = torch.randn(inputs["actions"].shape, generator=gen).to(cfg.device)
        # Deterministic sweep over t rather than a fixed constant: a single t
        # would only measure one slice of the velocity field, and the slices
        # differ enormously in difficulty.
        t = torch.linspace(0.05, 0.95, inputs["actions"].shape[0], device=cfg.device)
        _, info = policy.forward(**inputs, noise=noise, t=t)
        total += info["loss"]
        n += 1
    policy.train()
    return total / max(n, 1)


def main():
    cfg, args = parse_args()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "build": args.build}, indent=2))

    print(
        f"[train] build={args.build} device={cfg.device} amp={cfg.use_amp} "
        f"bs={cfg.batch_size}x{cfg.grad_accum_steps}={cfg.effective_batch_size} "
        f"steps={cfg.n_steps} chunk={cfg.chunk_size} language={cfg.language_mode}"
    )

    train_loader, val_loader, stats = make_loaders(cfg)
    tokenizer = make_tokenizer(cfg)
    normalizer = Normalizer(stats, cfg.state_dim, cfg.action_dim).to(cfg.device)
    policy = build_policy(cfg, args.build, args.pretrained).to(cfg.device)
    print("[train] " + policy.param_summary().replace("\n", "\n[train] "))

    optimizer = build_optimizer(policy, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    # cfg.__post_init__ forces use_amp off anywhere but CUDA, so device_type is
    # only ever "cuda" when the context is actually enabled.
    amp_kwargs = dict(
        device_type="cuda" if cfg.device == "cuda" else "cpu",
        dtype=torch.bfloat16,
        enabled=cfg.use_amp,
    )

    start_step = 0
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(blob["model"], strict=False)
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        start_step = blob["step"]
        print(f"[train] resumed from {args.resume} at step {start_step}")

    log_path = out / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as fp:
            csv.writer(fp).writerow(["step", "loss", "val_loss", "lr", "grad_norm", "sec_per_step"])

    best_val = float("inf")
    data_iter = cycle(train_loader)
    policy.train()
    t0 = time.time()

    for step in range(start_step + 1, cfg.n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = next(data_iter)
            inputs = build_model_inputs(batch, cfg, tokenizer, normalizer, cfg.device)
            with torch.autocast(**amp_kwargs):
                loss, info = policy.forward(**inputs)
            # Divide by the accumulation count so the gradient matches what a
            # single bs=64 step would have produced, not 4x it.
            (loss / cfg.grad_accum_steps).backward()
            running += info["loss"] / cfg.grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], cfg.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()

        val = ""
        if step % cfg.val_freq == 0 or step == cfg.n_steps:
            val = evaluate(policy, val_loader, cfg, tokenizer, normalizer, cfg.val_batches)
            if val < best_val:
                best_val = val
                save_checkpoint(
                    out / "best.pt", step, policy, normalizer, cfg, args.build,
                    {"val_loss": val}, args.pretrained,
                )
                print(f"[train] step {step}: new best val_loss={val:.6f} -> best.pt")

        if step % cfg.log_freq == 0 or step == cfg.n_steps:
            sps = (time.time() - t0) / cfg.log_freq
            t0 = time.time()
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step}/{cfg.n_steps} loss={running:.6f} "
                f"val={val if val == '' else f'{val:.6f}'} lr={lr:.2e} "
                f"|g|={float(grad_norm):.2f} ({sps:.2f}s/it)"
            )
            with open(log_path, "a", newline="") as fp:
                csv.writer(fp).writerow(
                    [step, f"{running:.6f}", val, f"{lr:.3e}", f"{float(grad_norm):.3f}",
                     f"{sps:.3f}"]
                )

        if step % cfg.save_freq == 0:
            save_checkpoint(
                out / "last.pt", step, policy, normalizer, cfg, args.build,
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                args.pretrained,
            )

    save_checkpoint(
        out / "last.pt", cfg.n_steps, policy, normalizer, cfg, args.build,
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        args.pretrained,
    )
    print(f"[train] done. best val_loss={best_val:.6f}. checkpoints in {out}")


if __name__ == "__main__":
    main()
