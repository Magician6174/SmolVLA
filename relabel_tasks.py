"""Per-episode language instructions, derived from recorded scene metadata.

WHY THIS EXISTS
---------------
The on-disk dataset has `total_tasks: 1`. Every frame points at a single task
string, "pick up the object and place it in the bin". For ACT and Diffusion
Policy that was irrelevant -- neither reads the task text. For a VLA it is
fatal: SmolVLA feeds `[image tokens | language tokens | state token]` to the VLM
and the action expert cross-attends to the result. If the language is
byte-identical on every sample it carries zero episode-discriminative
information, the model learns from vision and state alone, and "language
conditioned policy" becomes an empty claim.

`scene_gen.py` randomises the object shape per episode and `recorder.py` already
writes it to `episodes_meta.jsonl` -- recorded at collection time, then never
plumbed to a policy. This module surfaces it as three instructions:

    box      -> "pick up the cube and place it in the bin"
    cylinder -> "pick up the cylinder and place it in the bin"
    sphere   -> "pick up the sphere and place it in the bin"

This is not label fabrication: the instruction is a true description of what the
demonstration does. (Contrast with something I deliberately do NOT synthesise,
e.g. "place it on the left side of the bin" -- placement side was never
recorded, so that would be invention.)

NON-DESTRUCTIVE BY DESIGN
-------------------------
`task_index` is a per-frame column inside the LeRobot data parquets. Rewriting
those risks exactly the class of corruption this dataset already suffered once.
So we do NOT touch the dataset. `dataset.py` calls `build_episode_instructions()`
once at startup and injects the string in `__getitem__`. The on-disk dataset
stays byte-for-byte identical, and ACT / Diffusion Policy are unaffected.

THE INDEX ALIGNMENT GOTCHA (this is the whole reason this file is 200 lines)
---------------------------------------------------------------------------
There are THREE different integers in play and they are not interchangeable:

1. sidecar take number -- `episodes_meta.jsonl` logs *takes*, not saved
   episodes. 205 rows for 200 saved episodes, because cancelled takes are
   logged too, and its `episode` field is 1-based and *repeats* on a cancel
   (take 5 appears once as success and three times as cancelled). So this field
   is unusable directly; what is usable is the 0-based ordinal among
   `outcome == "success"` rows, which equals the original episode_index.

2. original `episode_index` -- what the recorder wrote into the parquets. In
   this dataset it is NOT dense: `meta/episodes` has 184 rows carrying values
   0..199 with 16 gaps ({26..34}, {105..111}) from two interrupted resume
   sessions that lost buffered metadata. `data/` still holds all 200 episodes'
   frames, so 4,252 of them are orphans with no metadata row.

3. episodes-table ROW POSITION 0..183 -- what lerobot's reader actually uses.
   `dataset_reader.get_item()` reads `ep_idx = row["episode_index"]` (a *value*)
   and then does `self._meta.episodes[ep_idx]`, which is a `datasets.Dataset`
   and therefore indexes POSITIONALLY. On a gappy dataset those two disagree
   for every episode after the first gap.

This module keys everything on (3), the row position, because that is the only
basis that survives `reindex_dataset.py`: reindexing preserves row order and
renumbers `episode_index` to equal the position. The mapping chain is

    position p -> episodes_table.episode_index[p] -> sidecar success ordinal
              -> object_type -> shape -> instruction

and if `.reindex_backup/` exists we read the *pre-reindex* episodes table from
it, because after reindexing the original episode_index is gone and the chain
would silently collapse to the identity (i.e. wrong by up to 16 positions).

Counts matching is weak evidence, not proof. `--verify` is the actual gate: it
decodes the first frame of a spread of positions and renders a contact sheet
with the predicted instruction overlaid, so the shape can be eyeballed. It
decodes straight from the video using the row's own chunk/file/from_timestamp
rather than going through `LeRobotDataset`, precisely so that the known
positional bug in the reader cannot mask (or fake) a misalignment here.

Usage:
    python relabel_tasks.py --data_root data/panda_pick_place
    python relabel_tasks.py --data_root data/panda_pick_place --verify 9
    python relabel_tasks.py --data_root data/panda_pick_place --dump task_map.json
"""
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# MuJoCo mjtGeom enum values, as written by scene_gen.py into the sidecar.
# (mjGEOM_SPHERE=2, mjGEOM_CYLINDER=5, mjGEOM_BOX=6)
GEOM_NAMES = {2: "sphere", 5: "cylinder", 6: "box"}

# The noun used in the instruction. "cube" rather than "box" because the geom is
# near-cubic and "box" collides with the bin ("place it in the bin") -- an
# avoidable ambiguity in a sentence that also names a container.
NOUNS = {"sphere": "sphere", "cylinder": "cylinder", "box": "cube"}

TEMPLATE = "pick up the {noun} and place it in the bin"

# The single string currently baked into meta/tasks.parquet, used as the
# "no shape information" control in the ablation.
GENERIC_INSTRUCTION = "pick up the object and place it in the bin"


def instruction_for(shape: str) -> str:
    return TEMPLATE.format(noun=NOUNS[shape])


def all_instructions() -> list[str]:
    """The 3 instructions, in a stable order (for ablation cycling)."""
    return [instruction_for(s) for s in ("box", "cylinder", "sphere")]


def _sidecar_shapes(root: Path) -> list[str]:
    """Shapes of the successful takes, in file order.

    Index into this list with the ORIGINAL episode_index (see gotcha 1 above).
    """
    sidecar = root / "episodes_meta.jsonl"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{sidecar} not found. The language relabel needs the recorder sidecar; "
            "without it there is no per-episode shape and the VLA would train on a "
            "constant instruction."
        )
    shapes = []
    for line in sidecar.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("outcome") != "success":
            continue  # cancelled takes were never written to the dataset
        otype = rec["object_type"]
        if otype not in GEOM_NAMES:
            raise ValueError(
                f"sidecar take {rec.get('episode')} has unknown object_type={otype}; "
                f"expected one of {sorted(GEOM_NAMES)}"
            )
        shapes.append(GEOM_NAMES[otype])
    return shapes


def _episodes_table_dir(root: Path) -> Path:
    """Where to read the episodes table from.

    Prefers `.reindex_backup/` when present: reindexing overwrites the original
    (gappy) episode_index with the dense position, which would destroy the only
    link back to the sidecar. Reading the backup keeps the map reproducible
    after a reindex instead of silently degenerating to the identity.
    """
    backup = root / ".reindex_backup" / "meta" / "episodes"
    return backup if backup.is_dir() else root / "meta" / "episodes"


def _original_episode_indices(root: Path) -> tuple[list[int], Path]:
    """Original `episode_index` per episodes-table row position."""
    import pyarrow.parquet as pq

    src = _episodes_table_dir(root)
    files = sorted(glob.glob(str(src / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no episodes parquet under {src}")
    out: list[int] = []
    for f in files:
        out += pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist()
    if out != sorted(out):
        # Path order is chunk-000/file-000, file-001, ... which is also global
        # episode order. If that ever stops holding, positions would be scrambled.
        raise ValueError(
            f"episodes table rows under {src} are not in ascending episode_index order; "
            "the position -> episode_index mapping cannot be trusted."
        )
    return out, src


def build_episode_instructions(data_root: str | Path) -> tuple[dict[int, str], dict[int, str]]:
    """Return (position -> instruction, position -> shape name).

    Keys are episodes-table row positions 0..N-1, which is what
    `LeRobotDataset` uses for per-episode metadata and what `episode_index`
    becomes after `reindex_dataset.py`. See the module docstring.
    """
    root = Path(data_root)
    shapes_by_orig = _sidecar_shapes(root)
    orig, src = _original_episode_indices(root)

    live = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if live:
        import pyarrow.parquet as pq

        n_live = sum(pq.read_metadata(f).num_rows for f in live)
        if n_live != len(orig):
            raise ValueError(
                f"alignment check FAILED: {len(orig)} rows in {src} but {n_live} rows in the "
                "live meta/episodes. The position -> instruction map would be off."
            )

    hi = max(orig)
    if hi >= len(shapes_by_orig):
        raise ValueError(
            f"alignment check FAILED: episodes table references original episode_index {hi} "
            f"but the sidecar only has {len(shapes_by_orig)} successful takes. Resolve before "
            "training -- silently mis-aligned language is worse than a crash."
        )

    instructions, shapes = {}, {}
    for pos, ep in enumerate(orig):
        shape = shapes_by_orig[ep]
        shapes[pos] = shape
        instructions[pos] = instruction_for(shape)
    return instructions, shapes


def summarise(instructions: dict[int, str], shapes: dict[int, str], data_root: str | Path) -> None:
    root = Path(data_root)
    orig, src = _original_episode_indices(root)
    gaps = sorted(set(range(max(orig) + 1)) - set(orig))
    counts = Counter(shapes.values())
    n = len(shapes)
    print(f"[relabel] {n} episode positions mapped to {len(set(instructions.values()))} instructions")
    print(f"[relabel] episodes table: {src.relative_to(root)}  "
          f"original episode_index 0..{max(orig)}"
          + (f", {len(gaps)} MISSING: {gaps}" if gaps else ", dense"))
    for shape in ("box", "cylinder", "sphere"):
        c = counts.get(shape, 0)
        print(f"  {shape:9s} {c:4d} eps ({c / max(n, 1):5.1%})  \"{instruction_for(shape)}\"")
    # Balance matters: a skewed split would confound the language ablation with
    # a class prior, so flag it rather than discover it in the results.
    if n:
        lo, hi = min(counts.values()), max(counts.values())
        if hi > 1.5 * lo:
            print(f"  WARNING: imbalanced ({lo}..{hi}); the ablation will be confounded "
                  "by an instruction prior.")
        else:
            print(f"  balance OK (min {lo}, max {hi}) -- ablation is not prior-confounded")


def _first_frame(root: Path, position: int, camera: str):
    """Decode the first frame of an episodes-table row, bypassing LeRobotDataset.

    `dataset_reader` looks the episode's video chunk/file/offset up POSITIONALLY
    while keying off the row's `episode_index` VALUE, which is exactly the bug
    we are guarding against. Reading the row ourselves means the gate tests the
    mapping, not the reader.
    """
    import pyarrow.parquet as pq
    from lerobot.datasets.video_utils import decode_video_frames

    src = _episodes_table_dir(root)
    files = sorted(glob.glob(str(src / "**" / "*.parquet"), recursive=True))
    key = f"observation.images.{camera}"
    cols = [f"videos/{key}/chunk_index", f"videos/{key}/file_index",
            f"videos/{key}/from_timestamp"]
    seen = 0
    for f in files:
        t = pq.read_table(f, columns=cols)
        if seen + t.num_rows > position:
            r = position - seen
            chunk = t.column(cols[0])[r].as_py()
            fidx = t.column(cols[1])[r].as_py()
            ts = t.column(cols[2])[r].as_py()
            break
        seen += t.num_rows
    else:
        raise IndexError(f"position {position} beyond {seen} episodes-table rows")

    # info.json video_path template, resolved against the live root (the backup
    # never copies videos -- reindex_dataset.py leaves them untouched).
    path = root / f"videos/{key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    frames = decode_video_frames(path, [ts], tolerance_s=1.0 / 30 + 1e-4)
    return frames[0], path, ts


def verify(data_root: str, n_show: int, out_path: str, camera: str = "front") -> None:
    """THE ALIGNMENT GATE.

    Decode the first frame of `n_show` positions spread across the full range
    and render a labelled contact sheet. If a tile's rendered object does not
    match the caption, the position -> instruction mapping is off and nothing
    downstream is trustworthy.
    """
    root = Path(data_root)
    instructions, shapes = build_episode_instructions(root)
    orig, _ = _original_episode_indices(root)
    n = len(shapes)
    # Spread across the range rather than taking the first N: an off-by-N
    # introduced by a mid-dataset gap only shows up after the gap.
    picks = sorted({int(round(i * (n - 1) / max(n_show - 1, 1))) for i in range(n_show)})

    tiles, captions = [], []
    for pos in picks:
        img, path, ts = _first_frame(root, pos, camera)
        arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        tiles.append(arr)
        captions.append(f"pos {pos} (orig ep {orig[pos]}): {shapes[pos]}")
        print(f"  pos {pos:3d} orig_ep {orig[pos]:3d} -> {shapes[pos]:9s} "
              f"| {path.name} @ {ts:.2f}s")

    _contact_sheet(tiles, captions, out_path)
    print(f"\n[verify] wrote {out_path}")
    print("[verify] GATE: open it and confirm every rendered object matches its caption.")
    print("[verify] If any tile disagrees, the position -> instruction mapping is wrong.")


def _contact_sheet(tiles, captions, out_path, cols: int = 3, tile_w: int = 320) -> None:
    """Assemble labelled tiles into one PNG. Uses PIL only (no matplotlib)."""
    from PIL import Image, ImageDraw

    scaled = []
    for arr in tiles:
        im = Image.fromarray(arr)
        h = int(im.height * tile_w / im.width)
        scaled.append(im.resize((tile_w, h), Image.BILINEAR))
    tw, th = scaled[0].size
    bar = 22  # caption strip height
    rows = (len(scaled) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + bar)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (im, cap) in enumerate(zip(scaled, captions)):
        r, c = divmod(i, cols)
        x, y = c * tw, r * (th + bar)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + th + 5), cap, fill=(255, 220, 90))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/panda_pick_place")
    ap.add_argument("--verify", type=int, default=0,
                    help="render a contact sheet of N episodes to eyeball alignment")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--out", default="task_alignment_check.png")
    ap.add_argument("--dump", default=None,
                    help="optional path to write the position -> instruction map as JSON")
    args = ap.parse_args()

    instructions, shapes = build_episode_instructions(args.data_root)
    summarise(instructions, shapes, args.data_root)

    if args.dump:
        orig, _ = _original_episode_indices(Path(args.data_root))
        Path(args.dump).write_text(json.dumps({
            "key": "episodes-table row position (== episode_index after reindex_dataset.py)",
            "original_episode_index": orig,
            "instructions": {str(k): v for k, v in instructions.items()},
            "shapes": {str(k): v for k, v in shapes.items()},
        }, indent=2))
        print(f"[relabel] wrote map to {args.dump}")

    if args.verify:
        verify(args.data_root, args.verify, args.out, camera=args.camera)


if __name__ == "__main__":
    main()
