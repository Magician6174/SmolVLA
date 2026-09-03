"""Closed-loop evaluation of a trained SmolVLA policy in MuJoCo (mac / glfw).

The success rate this script prints is the only number worth quoting. Validation
flow-matching loss measures how well the network regresses a velocity field; it
says almost nothing about whether the gripper closes on the object, and our ACT
and Diffusion runs both demonstrated the gap (Diffusion had the lower training
loss of the two and a third of the success rate).

    cd Panda
    KMP_DUPLICATE_LIB_OK=TRUE MUJOCO_GL=glfw \
      ~/miniconda3/envs/python_robotics/bin/python SmolVLA/rollout.py \
      --checkpoint SmolVLA/checkpoints/best.pt --episodes 200 --no_render

THE LANGUAGE ABLATION IS THE POINT OF THIS FILE
-----------------------------------------------
`scene_gen` samples the object shape per episode and reports it as
`info["object_type"]`, so at rollout we know the ground truth and can hand the
policy any instruction we like while the scene stays fixed. `--sweep_language`
runs all four conditions over the SAME seed sequence, which makes them paired:
every condition sees an identical set of 200 scenes, so a difference in success
rate cannot be scene luck.

    correct   the true shape
    mismatch  a different shape, via the fixed cycle box->cylinder->sphere->box
    empty     no instruction at all
    generic   "pick up the object..." -- well-formed, but names no shape

Reading the result:
    correct > mismatch ~ empty   the policy genuinely reads the noun
    correct ~ mismatch ~ empty   vision alone disambiguates; the language stream
                                 is decoration and calling this a VLA is a stretch
    correct ~ mismatch > empty   language works only as a "task is active" gate

This is worth stating plainly because our task is nearly the third case by
construction: all three shapes go in the same bin, so the *goal* never depends on
the noun -- only the grasp geometry does. A null result here is an honest finding
about the task, not a failed experiment.

RECEDING HORIZON
----------------
The policy emits 50 actions per forward pass. `n_action_steps` decides how many
are executed before re-planning: 1 = re-observe every frame (the paper's
simulation setting, and the regime our ACT/DP numbers were measured in), 50 =
fully open-loop (lerobot's default). At n=1 every frame costs a full VLM prefix
pass plus 10 Euler steps, so this is by far the slowest of our three policies to
evaluate. There is no temporal ensembling: ACT needed it because consecutive
chunks disagreed, and testing whether flow matching needs it too is a separate
experiment (the `--n_action_steps` sweep is the cheap version of that question).
"""
import argparse
import os
import sys
from collections import deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PANDA)

from dataset import GENERIC_INSTRUCTION, INSTRUCTION_OF_SHAPE, SHAPE_CYCLE, make_tokenizer
from train import load_policy

import control as C
from scene_gen import SceneManager

# MuJoCo mjtGeom codes, as written into info["object_type"] by scene_gen.
GEOM_NAMES = {2: "sphere", 5: "cylinder", 6: "box"}
LANGUAGE_MODES = ("correct", "mismatch", "empty", "generic")


def instruction_for(shape: str, mode: str) -> str:
    """Same mapping as training (dataset.TaskRelabeller), driven by the live scene."""
    if mode == "empty":
        return ""
    if mode == "generic":
        return GENERIC_INSTRUCTION
    if mode == "mismatch":
        shape = SHAPE_CYCLE[shape]
    elif mode != "correct":
        raise ValueError(f"unknown language mode {mode!r}")
    return INSTRUCTION_OF_SHAPE[shape]


class PolicyRunner:
    """Receding-horizon wrapper: predict 50, execute n_action_steps, repeat.

    The queue lives here rather than inside the model because `model.py` is
    deliberately a pure function of its inputs -- no hidden state, so
    `parity_test.py` can compare it against lerobot's implementation without
    having to reset anything.
    """

    def __init__(self, policy, normalizer, cfg, tokenizer, n_action_steps: int | None = None):
        self.policy, self.normalizer, self.cfg, self.tokenizer = policy, normalizer, cfg, tokenizer
        self.n_action_steps = n_action_steps or cfg.n_action_steps
        self.queue: deque = deque()
        self.instruction = ""
        self.n_forwards = 0

    def reset(self, instruction: str) -> None:
        self.queue.clear()
        self.instruction = instruction

    def _observe(self, viewer) -> dict:
        """Live MuJoCo frame -> the same batch dict shape the DataLoader produces.

        State is `[qpos[:7], qpos[7]]`: seven arm joints plus one finger position.
        Identical to what the recorder wrote, which is the whole reason a policy
        trained offline can be driven here at all.
        """
        state = np.concatenate([C.data.qpos[:7], [C.data.qpos[7]]]).astype(np.float32)
        batch = {"observation.state": torch.from_numpy(state)[None]}
        for cam in self.cfg.cameras:
            rgb = viewer.grab(cam)
            t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{cam}"] = t[None]
        return batch

    @torch.no_grad()
    def select_action(self, viewer) -> np.ndarray:
        if not self.queue:
            from dataset import build_model_inputs

            batch = self._observe(viewer)
            inputs = build_model_inputs(
                batch, self.cfg, self.tokenizer, self.normalizer, self.cfg.device,
                tasks=[self.instruction],
            )
            chunk = self.policy.sample_actions(**inputs)          # (1, 50, 32) normalised
            chunk = self.normalizer.unnormalize_action(chunk[0, :, : self.cfg.action_dim])
            self.n_forwards += 1
            for a in chunk[: self.n_action_steps]:
                self.queue.append(a.float().cpu().numpy())
        return self.queue.popleft()


def is_success(info: dict) -> bool:
    """Object resting inside the bin. Byte-identical criterion to ACT and DP."""
    bx, by, bin_z = info["bin_pose"]
    half = info["bin"]["half"]
    rim = bin_z + info["bin"]["wh"]
    obj = C.data.body("object").xpos
    in_xy = abs(obj[0] - bx) <= half and abs(obj[1] - by) <= half
    return bool(in_xy and obj[2] <= rim + 0.01 and obj[2] > 0.0)


def run_episode(runner, sm, rng, viewer, mode: str, max_steps: int, render: bool):
    info = C.reset_episode(sm, rng, viewer)
    shape = GEOM_NAMES[int(info["object_type"])]
    runner.reset(instruction_for(shape, mode))
    substeps = max(1, round((1.0 / runner.cfg.fps) / C.SIM_DT))

    for _ in range(max_steps):
        action = runner.select_action(viewer)
        C.data.ctrl[:7] = action[:7]
        C.data.ctrl[C.GRIPPER] = action[7]
        np.clip(C.data.ctrl, C.CTRL_RANGE[:, 0], C.CTRL_RANGE[:, 1], out=C.data.ctrl)
        for _ in range(substeps):
            mujoco.mj_step(C.model, C.data)
        if render:
            viewer.render()
        if is_success(info):
            return True, shape
    return False, shape


def evaluate_mode(runner, sm, viewer, mode: str, episodes: int, seed: int,
                  max_steps: int, render: bool) -> dict:
    """One condition, `episodes` scenes. The RNG is re-seeded per mode so every
    condition sees the identical sequence of scenes -- that is what makes the
    comparison paired rather than two independent samples."""
    rng = np.random.default_rng(seed)
    per_shape: dict[str, list[bool]] = {}
    results = []
    for ep in range(episodes):
        ok, shape = run_episode(runner, sm, rng, viewer, mode, max_steps, render)
        results.append(ok)
        per_shape.setdefault(shape, []).append(ok)
        print(
            f"  [{mode}] episode {ep + 1:3d}/{episodes}: {'SUCCESS' if ok else 'fail  '} "
            f"({shape:8s}) running {sum(results)}/{len(results)} = {np.mean(results):.0%}"
        )
    return {
        "mode": mode,
        "n": len(results),
        "success_rate": float(np.mean(results)),
        "per_shape": {k: (float(np.mean(v)), len(v)) for k, v in sorted(per_shape.items())},
        "forwards": runner.n_forwards,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="SmolVLA/checkpoints/best.pt")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=500, help="policy frames per episode (@fps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--language_mode", default="correct", choices=LANGUAGE_MODES)
    ap.add_argument("--sweep_language", action="store_true",
                    help="run all four language conditions over the same scenes")
    ap.add_argument("--n_action_steps", type=int, default=None,
                    help="override: 1 = re-observe every frame, 50 = open-loop chunk")
    ap.add_argument("--num_inference_steps", type=int, default=None, help="Euler steps")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    overrides = {}
    if args.num_inference_steps:
        overrides["num_inference_steps"] = args.num_inference_steps
    policy, normalizer, cfg = load_policy(args.checkpoint, device=device, cfg_overrides=overrides)
    tokenizer = make_tokenizer(cfg)
    runner = PolicyRunner(policy, normalizer, cfg, tokenizer, args.n_action_steps)

    print(
        f"[rollout] device={device} episodes={args.episodes} chunk={cfg.chunk_size} "
        f"n_action_steps={runner.n_action_steps} euler_steps={cfg.num_inference_steps}"
    )

    sm = SceneManager()
    viewer = C.Viewer(title="SmolVLA rollout")
    modes = LANGUAGE_MODES if args.sweep_language else (args.language_mode,)

    summaries = []
    for mode in modes:
        runner.n_forwards = 0
        summaries.append(
            evaluate_mode(runner, sm, viewer, mode, args.episodes, args.seed,
                          args.max_steps, not args.no_render)
        )
    viewer.close()

    print("\n[rollout] === summary ===")
    for s in summaries:
        shapes = "  ".join(f"{k}={r:.0%}({n})" for k, (r, n) in s["per_shape"].items())
        print(f"  {s['mode']:9s} {s['success_rate']:6.1%} of {s['n']}   {shapes}")
    if len(summaries) > 1:
        base = next(s["success_rate"] for s in summaries if s["mode"] == "correct")
        print("\n[rollout] deltas against 'correct' (paired: identical scenes):")
        for s in summaries:
            if s["mode"] != "correct":
                print(f"  {s['mode']:9s} {s['success_rate'] - base:+.1%}")


if __name__ == "__main__":
    main()
