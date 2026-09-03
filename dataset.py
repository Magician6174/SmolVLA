"""Data plumbing for SmolVLA: windows, language relabelling, normalisation.

Three jobs, in increasing order of how easy they are to get wrong.

1. WINDOWING. Much simpler than Diffusion Policy's. SmolVLA conditions on the
   CURRENT frame only (`n_obs_steps=1`), so there is no observation history to
   stack; the only window is the 50-step action chunk `[0 .. 49]` starting at the
   current frame. That matches lerobot's `SmolVLAConfig.action_delta_indices`.

2. LANGUAGE. The dataset on disk has `total_tasks: 1` -- every one of the 45,030
   frames carries the same string. Training a vision-language-action model on
   that teaches it nothing about language; the text stream would be a constant
   and the model would (correctly) learn to ignore it. `relabel_tasks.py`
   recovered the per-episode object shape from the recorder sidecar and wrote
   `task_map.json`; this module applies it, and supports the three controls that
   turn "we used a VLM" into an actual claim about language conditioning. Nothing
   on disk is modified -- the relabelling is a metadata lookup at sample time.

3. NORMALISATION IS MEAN/STD HERE, NOT MIN/MAX. Our Diffusion Policy uses
   min-max to [-1, 1] because DDPM's x_0 has to live in the range the noise
   schedule assumes. Flow matching has no such range, but it does have a scale
   requirement that points the other way: the path is `x_t = t*noise + (1-t)*A`
   with `noise ~ N(0, I)`, so the two endpoints are only commensurate if the
   actions are also roughly unit variance. Min-max normalised joint targets have
   std ~0.2-0.3, which would make the interpolation almost pure noise for most
   of t and hand the network a target (`noise - A`) dominated by the noise term.
   Mean/std it is -- which is also what lerobot's SmolVLA config specifies.

ORDER OF OPERATIONS FOR THE 8 -> 32 PADDING
-------------------------------------------
Normalise the 8 real dims, THEN zero-pad to 32. Doing it the other way round
would compute statistics over 24 constant-zero columns and, worse, would divide
them by a zero std. The pad columns must be exactly zero going in, and the loss
ignores them coming out (see `model.py`).
"""
import os

# Local-only dataset; never reach the Hub. Must be set before importing lerobot.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from config import SmolVLAConfig
from vlm_backbone import resize_with_pad

# The single string that is actually stored in the dataset's task table. It is
# also the "no shape information" control: a well-formed instruction that simply
# does not say which object to pick.
GENERIC_INSTRUCTION = "pick up the object and place it in the bin"

# A fixed cyclic derangement of the three shapes, used for the "mismatch"
# control. A cycle rather than a random draw for two reasons: every episode is
# guaranteed to get a WRONG instruction (a random draw would sometimes hit the
# right one), and the marginal distribution over the three instructions is
# unchanged, so the only thing that differs from "correct" is the pairing. If
# accuracy holds up under this, the policy was not reading the noun.
SHAPE_CYCLE = {"box": "cylinder", "cylinder": "sphere", "sphere": "box"}

INSTRUCTION_OF_SHAPE = {
    # "cube" not "box": the word "box" collides with "bin" in the same sentence,
    # and "cube" is the more common token in image-text pretraining corpora.
    "box": "pick up the cube and place it in the bin",
    "cylinder": "pick up the cylinder and place it in the bin",
    "sphere": "pick up the sphere and place it in the bin",
}


class TaskRelabeller:
    """episode_index -> instruction string, under one of four language modes.

    Keyed on `episode_index` as returned by the dataset, which is only the same
    thing as the `task_map.json` key if the episodes table is DENSE (values equal
    positions). That held before this dataset was reindexed and it holds after,
    but it did NOT hold in between, and the failure is silent -- so
    `assert_dense_episodes` is called on construction rather than trusted.
    """

    def __init__(self, cfg: SmolVLAConfig, map_path: str | Path | None = None):
        self.cfg = cfg
        path = Path(map_path or Path(__file__).parent / "task_map.json")
        blob = json.loads(path.read_text())
        self.shapes: dict[int, str] = {int(k): v for k, v in blob["shapes"].items()}
        self.instructions: dict[int, str] = {int(k): v for k, v in blob["instructions"].items()}
        self.original_episode_index: list[int] = blob["original_episode_index"]

    def __len__(self) -> int:
        return len(self.shapes)

    def instruction(self, episode_index: int, mode: str | None = None) -> str:
        mode = mode or self.cfg.language_mode
        if not self.cfg.use_relabelled_instructions:
            return GENERIC_INSTRUCTION
        if mode == "empty":
            return ""
        if mode == "generic":
            return GENERIC_INSTRUCTION
        shape = self.shapes[int(episode_index)]
        if mode == "mismatch":
            shape = SHAPE_CYCLE[shape]
        elif mode != "correct":
            raise ValueError(f"unknown language_mode {mode!r}")
        return INSTRUCTION_OF_SHAPE[shape]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for s in self.shapes.values():
            counts[s] = counts.get(s, 0) + 1
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        return f"{len(self)} episodes, mode={self.cfg.language_mode}, shapes: {parts}"


def assert_dense_episodes(meta: LeRobotDatasetMetadata) -> int:
    """Fail loudly if the episodes table is gappy.

    This dataset was recorded across interrupted sessions and lost the metadata
    for 16 episodes, leaving an episodes table whose 184 rows carried
    `episode_index` values spread over 0..199. lerobot's reader indexes that
    table POSITIONALLY with a VALUE (`self._meta.episodes[row["episode_index"]]`),
    so every episode after the first gap was served the wrong video segment --
    frames from one demonstration paired with the actions of another. Training
    still converges, on garbage. `ACT/reindex_dataset.py` is the fix.

    The check is one line and it guards every consumer of `task_map.json`.
    """
    values = list(meta.episodes["episode_index"])
    n = len(values)
    if values != list(range(n)):
        missing = sorted(set(range(max(values) + 1)) - set(values))
        raise RuntimeError(
            f"episodes table is not dense: {n} rows spanning 0..{max(values)}, "
            f"missing {missing[:12]}{'...' if len(missing) > 12 else ''}. "
            "lerobot indexes this table positionally with an episode_index VALUE, "
            "so every episode after the first gap decodes the wrong video segment. "
            "Run `python ACT/reindex_dataset.py <data_root>` before training."
        )
    return n


# --- normalisation ------------------------------------------------------------
def _to_tensor(x) -> Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).flatten()


class Normalizer(nn.Module):
    """(x - mean) / std for state and action, plus the inverse for actions.

    Registered as buffers so the statistics travel with the checkpoint. That is
    not a convenience: a policy whose weights are restored with different
    normalisation constants than it was trained with produces plausible-looking
    joint targets that are quietly offset, which at rollout looks like a
    modelling failure rather than a bookkeeping one.
    """

    def __init__(self, stats: dict, state_dim: int, action_dim: int):
        super().__init__()
        for key, dim in (("observation.state", state_dim), ("action", action_dim)):
            mean, std = _to_tensor(stats[key]["mean"]), _to_tensor(stats[key]["std"])
            if mean.numel() != dim:
                raise ValueError(f"{key}: stats have {mean.numel()} dims, config says {dim}")
            self.register_buffer(f"{key.split('.')[-1]}_mean", mean)
            # A joint that never moved in the whole dataset has std 0. Clamping
            # keeps it at its mean instead of producing inf.
            self.register_buffer(f"{key.split('.')[-1]}_std", std.clamp_min(1e-8))

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Normalizer":
        """Rebuild from saved buffers alone, without needing the dataset.

        Rollout loads a checkpoint on a machine that may not have the dataset
        mounted at all, so the normaliser has to be reconstructible from the
        checkpoint's four tensors.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        for k, v in sd.items():
            obj.register_buffer(k, torch.as_tensor(v))
        return obj

    def normalize_state(self, x: Tensor) -> Tensor:
        return (x - self.state_mean) / self.state_std

    def normalize_action(self, x: Tensor) -> Tensor:
        return (x - self.action_mean) / self.action_std

    def unnormalize_action(self, x: Tensor) -> Tensor:
        return x * self.action_std + self.action_mean


def pad_to(x: Tensor, width: int) -> Tensor:
    """Zero-pad the last dim from `real` to `width` (8 -> 32). See module docstring."""
    if x.shape[-1] == width:
        return x
    if x.shape[-1] > width:
        raise ValueError(f"cannot pad {x.shape[-1]} down to {width}")
    pad = torch.zeros(*x.shape[:-1], width - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


# --- tokenisation -------------------------------------------------------------
def make_tokenizer(cfg: SmolVLAConfig):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(cfg.vlm_model_name).tokenizer


def tokenize_tasks(tokenizer, tasks: list[str], cfg: SmolVLAConfig) -> tuple[Tensor, Tensor]:
    """Tokenise instructions exactly the way lerobot's processor pipeline does.

    The trailing newline is not cosmetic. lerobot inserts a
    `NewLineTaskProcessorStep` before tokenisation (a PaliGemma-era convention
    carried over from openpi), so `smolvla_base` was pretrained on strings that
    all end in "\\n". Omitting it in the from-scratch build would be harmless on
    its own but would make the finetuning baseline see a different token
    sequence than its pretraining did, for no reason.

    `padding="longest"` (lerobot's default) keeps the prefix as short as the
    batch allows -- with our ~10-token instructions that is 11 tokens instead of
    a padded 48, which is 37 fewer prefix positions to attend over on every one
    of the 10 Euler steps.
    """
    tasks = [t if t.endswith("\n") else t + "\n" for t in tasks]
    out = tokenizer(
        tasks,
        padding=cfg.pad_language_to,
        padding_side="right",
        max_length=cfg.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return out["input_ids"], out["attention_mask"].bool()


# --- dataset ------------------------------------------------------------------
def _delta_timestamps(cfg: SmolVLAConfig) -> dict:
    """Only the action key gets a window: chunk_size steps from the current frame.

    Observation keys are deliberately absent. lerobot returns un-windowed keys
    with no leading time axis, so images arrive as (3, H, W) and state as
    (state_dim,) -- which is what we want, since `n_obs_steps=1` means there is
    no history to stack and a length-1 axis would only have to be squeezed away.
    """
    return {"action": [i / cfg.fps for i in range(cfg.chunk_size)]}


class _PrepareSample(torch.utils.data.Dataset):
    """Worker-side letterboxing and language lookup.

    Resizing here rather than in the model is the same /dev/shm fix the Diffusion
    loader needed: the DataLoader ships 512x512 tensors between processes instead
    of native 480x640 (which is, incidentally, a 15% reduction rather than an
    increase -- 786k floats against 922k).

    Images stay in [0, 1] here. The final `2x - 1` shift into SigLIP's expected
    range happens in `build_model_inputs`, so that rollout -- which never touches
    this class, because MuJoCo hands frames straight to the policy -- goes
    through exactly one shared preprocessing path.
    """

    def __init__(self, ds, cfg: SmolVLAConfig, relabeller: TaskRelabeller):
        self.ds = ds
        self.cfg = cfg
        self.relabeller = relabeller
        self.image_keys = [f"observation.images.{c}" for c in cfg.cameras]
        self.size = tuple(cfg.resize_imgs_with_padding)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        for k in self.image_keys:
            v = sample[k]
            if v.shape[-2:] != self.size:
                sample[k] = resize_with_pad(v[None], *self.size)[0]
        sample["task"] = self.relabeller.instruction(int(sample["episode_index"]))
        return sample


def episode_split(cfg: SmolVLAConfig) -> tuple[list[int], list[int]]:
    """Split by EPISODE, so validation frames come from unseen demonstrations.

    Splitting by frame would put frames 100 and 101 of the same demo on opposite
    sides of the split and report a validation loss that is essentially training
    loss. Same seed as the ACT and Diffusion runs, so all three policies are
    evaluated on the same held-out demos.
    """
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.data_root)
    n = assert_dense_episodes(meta)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * cfg.val_fraction)))
    return sorted(order[n_val:].tolist()), sorted(order[:n_val].tolist())


def make_datasets(cfg: SmolVLAConfig):
    train_eps, val_eps = episode_split(cfg)
    relabeller = TaskRelabeller(cfg)
    common = dict(root=cfg.data_root, delta_timestamps=_delta_timestamps(cfg))

    train_raw = LeRobotDataset(cfg.repo_id, episodes=train_eps, **common)
    val_raw = LeRobotDataset(cfg.repo_id, episodes=val_eps, **common)
    stats = train_raw.meta.stats
    print(f"[dataset] {relabeller.summary()}")
    print(
        f"[dataset] train: {len(train_eps)} eps / {len(train_raw)} frames | "
        f"val: {len(val_eps)} eps / {len(val_raw)} frames"
    )
    return (
        _PrepareSample(train_raw, cfg, relabeller),
        _PrepareSample(val_raw, cfg, relabeller),
        stats,
    )


def make_loaders(cfg: SmolVLAConfig):
    train_ds, val_ds, stats = make_datasets(cfg)
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,          # keeps the effective batch size honest under grad accumulation
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        common["prefetch_factor"] = cfg.prefetch_factor
    return (
        torch.utils.data.DataLoader(train_ds, shuffle=True, **common),
        torch.utils.data.DataLoader(val_ds, shuffle=False, **common),
        stats,
    )


# --- the one place the model's input contract lives ---------------------------
def build_model_inputs(
    batch: dict,
    cfg: SmolVLAConfig,
    tokenizer,
    normalizer: Normalizer,
    device: str,
    tasks: list[str] | None = None,
) -> dict:
    """Collated batch -> exactly the keyword arguments `SmolVLAFromScratch` wants.

    Used by train.py, the validation loop, rollout.py and model_lib.py, so that
    "what the model is fed" is defined once. `tasks` overrides the batch's own
    instructions, which is how the language ablation is run at evaluation time
    against a single fixed checkpoint (no retraining -- the comparison is about
    what the trained policy USES, not what it could have learned).
    """
    b = len(batch["observation.state"])
    images, img_masks = [], []
    for cam in cfg.cameras:
        img = batch[f"observation.images.{cam}"].to(device, non_blocking=True)
        if img.shape[-2:] != tuple(cfg.resize_imgs_with_padding):
            img = resize_with_pad(img, *cfg.resize_imgs_with_padding)
        images.append(img.float() * 2.0 - 1.0)     # [0,1] -> SigLIP's [-1,1]
        # All three cameras are always present in this dataset. The mask exists
        # because SmolVLA supports padding a variable camera set up to a fixed
        # slot count; kept so the plumbing matches lerobot's.
        img_masks.append(torch.ones(b, dtype=torch.bool, device=device))

    if tasks is None:
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks] * b
    lang_tokens, lang_masks = tokenize_tasks(tokenizer, list(tasks), cfg)

    state = batch["observation.state"].to(device).float()
    state = pad_to(normalizer.normalize_state(state), cfg.max_state_dim)

    out = dict(
        images=images,
        img_masks=img_masks,
        lang_tokens=lang_tokens.to(device),
        lang_masks=lang_masks.to(device),
        state=state,
    )
    if "action" in batch:
        actions = batch["action"].to(device).float()
        out["actions"] = pad_to(normalizer.normalize_action(actions), cfg.max_action_dim)
        pad = batch.get("action_is_pad")
        out["actions_is_pad"] = pad.to(device) if pad is not None else None
    return out
