# SmolVLA on the Panda pick-and-place task

A from-scratch reimplementation of SmolVLA's **flow-matching action expert**, plus a
finetune of the released `lerobot/smolvla_base` checkpoint, both trained and evaluated
on the same 184-episode MuJoCo Franka Panda dataset used by the ACT and Diffusion
Policy builds in this repo.

The point of the exercise is to understand vision-language-action models and the flow
matching objective by writing the parts that matter, not to beat a benchmark. The
frozen VLM is loaded from Hugging Face (`SmolVLM2-500M-Video-Instruct`); everything
downstream of it -- the expert transformer, the interleaved self/cross attention, the
flow-matching loss, the Euler sampler, the data pipeline, the training loop and the
closed-loop evaluator -- is written here.

Theory writeup: **[FLOW_MATCHING.md](FLOW_MATCHING.md)** (474 lines) derives the
objective, the probability-path construction, why the network predicts a velocity
rather than a noise, how this differs from DDPM, and where the paper's own tables
contradict its prose.

---

## 1. What is actually being compared

Three builds, chosen so that "is my implementation faithful?" and "does large-scale
robot pretraining transfer?" are separate questions with separate answers.

| Build | Command | Frozen | Trainable | Answers |
|---|---|---|---|---|
| `scratch` | `--build scratch` | SmolVLM2 (303M) | my expert (99.9M) | Can a from-scratch expert learn this task at all? |
| `finetune --pretrained none` | `--build finetune --pretrained none` | SmolVLM2 (350M) | lerobot's expert (99.9M) | Is my expert equivalent to lerobot's *architecture*, independent of weights? |
| `finetune` | `--build finetune` | SmolVLM2 (350M) | `smolvla_base` expert (99.9M) | What does pretraining on 481 real-robot datasets buy on a sim task it has never seen? |

The middle row is the control that most reimplementations skip. Without it, a gap
between rows 1 and 3 is unattributable: it could be my code, or it could be the
pretrained weights.

Both builds report **99,880,992 trainable parameters**, verified equal by
`parity_test.py` test 10. The from-scratch build totals 402,737,376 parameters
(302,856,384 frozen); lerobot's totals 450,046,176, the difference being the unused
`lm_head` discussed in section 6.

---

## 2. Files

| File | Role |
|---|---|
| `FLOW_MATCHING.md` | The theory. Read this first. |
| `config.py` | One dataclass, every hyperparameter, each non-obvious value annotated with where it comes from (paper table, lerobot source line, or a decision made here). |
| `vlm_backbone.py` | `FrozenSmolVLM`: loads SmolVLM2, truncates to 16 layers, and runs the text stack **by hand** so the per-layer K/V can be captured for the expert to cross-attend into. Also `apply_rope`, `make_att_2d_masks`, `resize_with_pad`, `ensure_attendable`. |
| `model.py` | `SmolVLAFromScratch`: the action expert, the interleaved FULL/CROSS attention lockstep, the flow-matching loss, and the Euler sampler. |
| `model_lib.py` | `SmolVLAFinetune`: wraps lerobot's `VLAFlowMatching` directly (bypassing `SmolVLAPolicy`'s processor pipeline, which `dataset.py` already replicates) so the two builds are trained by the identical loop and reduced to a bit-identical scalar. |
| `dataset.py` | Loading, the 3-way language relabel, MEAN_STD normalisation, letterboxing, tokenisation, the train/val split, and `build_model_inputs` -- the single place the model's input contract lives. |
| `train.py` | The loop: grad accumulation 16x4, cosine LR with warmup, bf16 autocast, deterministic paired validation. |
| `rollout.py` | Closed-loop MuJoCo evaluation with a receding-horizon action queue, and the paired language ablation. |
| `relabel_tasks.py` | Derives per-episode instructions from the `object_type` sidecar and writes `task_map.json`. |
| `parity_test.py` | 10 numerical tests against lerobot. **10/10 passing.** |
| `task_map.json` | The frozen relabel: 184 episodes, cube 62 / cylinder 67 / sphere 55. |

---

## 3. The dataset bug that had to be fixed first

`ACT/data/panda_pick_place_rgb_only` shipped with a **sparse** episodes table: 184 rows
carrying `episode_index` values spread over 0..199 with 16 values missing. lerobot's
`dataset_reader.get_item()` does `self._meta.episodes[row["episode_index"]]` -- it
indexes a `datasets.Dataset` **positionally using a value**. With a sparse table,
158 of 184 episodes therefore decoded video frames from the *wrong episode*.

This is silent. Training runs, loss descends, and the policy learns to map one
episode's images onto another episode's actions.

Fixed by `ACT/reindex_dataset.py` (dense 0..183, 45,030 frames, backup in
`.reindex_backup/`), verified with mean absolute reader-vs-truth error of `0.000000`,
and guarded permanently: `dataset.assert_dense_episodes()` refuses to build a loader
on a sparse table and tells you which script to run.

---

## 4. Architecture, in the order data flows

```
3 cameras (480x640) -> letterbox 512x512, [-1,1] -> SigLIP -> 64 tokens each = 192
instruction (~11 tokens, padded)                 -> SmolLM2 embeddings  =  48
proprioception (8 dims, zero-padded to 32)       -> state_proj          =   1
                                                                          ----
                                            frozen VLM prefix (16 layers) 241 tokens
                                                     |
                                        per-layer keys/values (cached once)
                                                     v
noise chunk (50 x 32) + time embedding -> action expert (16 layers, d=720) -> velocity
```

Details worth knowing, all verified against the source:

- **Lockstep, not two towers.** Expert layer *i* reads the VLM's layer-*i* K/V.
  `layer_idx % 2 == 0` gives a FULL attention over `[prefix keys | action keys]`;
  odd layers CROSS-attend to the prefix only.
- **The action chunk is causal.** `att_masks` is all ones over the 50 action tokens,
  so token *j* cannot see token *j+1*. Ablation switch: `causal_chunk=False`.
- **The prefix is one bidirectional block that cannot see the state token**
  (`att_masks = [0]*192 + [0]*48 + [1]`). That is what makes the prefix K/V cacheable
  across all 10 Euler steps -- the expensive half runs once per action chunk, not once
  per integration step.
- **RoPE positions reset on cross layers.** FULL layers give action tokens positions
  `241..290`; CROSS layers reset them to `0..49`. Switch: `reset_cross_positions`.
- **sqrt(d) embedding scale** (~31x) on image and language embeddings, the
  PaliGemma/openpi convention. Llama does not do this. Switch: `scale_embeddings`.
- **Expert width** `int(960 * 0.75) = 720`, FFN `2048`. On CROSS layers the k/v
  projections are `320 -> 320` adapters, not `720 -> 320`.

### Flow matching, in four lines

```
x_t = t * noise + (1 - t) * actions          # t=1 is NOISE, t=0 is data
u_t = noise - actions                        # the target velocity, constant per sample
loss = MSE(v_theta(x_t, t, prefix), u_t)     # masked back to the real 8 dims
sample: x = noise; for k in range(10): t = 1 + k*dt; x += dt * v_theta(x, t)   # dt = -1/10
```

The sign convention here is the **code's**, not the paper's; paper section 3.1 is
inconsistent with the released implementation. `parity_test.py` test 7 pins the
convention down without any trained weights: substitute the ideal velocity
`noise - actions` for the network, integrate, and demand the exact ground-truth chunk
back. Only one combination of time direction, `dt` sign and step count satisfies that,
and it is exact for any N (a straight line is integrated exactly by Euler).

This matters more than it sounds. A sign error trains to a perfectly respectable loss
and then emits garbage at rollout, because the integrator walks the wrong way along a
field it learned correctly.

---

## 5. The language ablation

The recorded dataset has `total_tasks: 1`: every episode carries the identical string
`"pick up the object and place it in the bin"`. Language would have been dead weight,
and a VLA with dead language is not a VLA.

`relabel_tasks.py` fixes this with a **metadata-only** relabel -- no re-collection.
Each episode has an `object_type` sidecar (MuJoCo `mjtGeom`: 2 sphere, 5 cylinder,
6 box), so three instructions can be derived and written into the tasks table:

- box -> `"pick up the cube and place it in the bin"`
- cylinder -> `"pick up the cylinder and place it in the bin"`
- sphere -> `"pick up the sphere and place it in the bin"`

("cube", not "box", so the token does not collide with "bin".)

Because `scene_gen` reports `info["object_type"]` at evaluation time too, the ablation
runs **closed-loop**, not merely as a validation loss. Four paired conditions, with the
RNG re-seeded per condition so all four see byte-identical scenes:

| Mode | Instruction given | Tests |
|---|---|---|
| `correct` | matches the object | the intended behaviour |
| `mismatch` | a fixed derangement (box->cylinder->sphere->box) | does a *wrong* instruction hurt? |
| `generic` | the original single string | does *any* shape information help? |
| `empty` | `""` | does the language channel matter at all? |

`--sweep_language` runs all four.

**Reading the result honestly.** `correct > mismatch ~ generic` means the policy is
genuinely language-conditioned. `correct ~ mismatch ~ generic` means vision alone
disambiguates. On this task the second outcome is the *expected* one by construction:
all three shapes go into the same bin, so the instruction is largely redundant with
what the wrist camera already sees. A null result here is an honest finding about the
task, not a failed experiment -- and it is worth having measured rather than assumed.


### Three findings that came out of writing these tests

**1. lerobot applies the wrong RoPE theta.** `SmolVLM2-500M` was pretrained with
`rope_theta = 100_000`; `smolvlm_with_expert.py` hardcodes `10_000`. In transformers
>= 5 the value moved to `text_config.rope_parameters["rope_theta"]` and is no longer a
top-level attribute, which is presumably how the bug survived -- a naive lookup returns
`None`. Measured cost: `apply_rope` outputs differ by up to **7.495**, and 4 layers of
frozen text encoder diverge by **2.69** in hidden state.

This project's default is deliberately **lerobot's 10,000**, not the correct value.
Using 100k for the from-scratch build and 10k for the finetune would confound the
primary comparison with a conditioning difference. `rope_theta="checkpoint"` is
available as a separate, later ablation.

**2. The advertised "450M" model really runs as ~403M.** lerobot's build carries a
frozen, untied `lm_head` of `49,280 x 960 = 47.31M` parameters that is never called --
the VLM is used as an encoder, so no logits are ever produced. `model.py` drops it.
The 450.0M figure in the paper is reproduced exactly by lerobot's build, so the number
is not wrong, just not the number of parameters that do any work.

**3. Fully-masked attention rows are a latent NaN.** A padding query row that attends
to nothing is a softmax over all `-inf`. PyTorch 2.10's CPU math backend happens to
return 0, but the CUDA flash and mem-efficient kernels make no such promise, and a
single NaN token poisons the whole batch because `0 * NaN = NaN`. `ensure_attendable`
gives every empty query row a self-attention diagonal, which is harmless because padded
rows are masked as *keys* everywhere, so nothing reads them. It is deliberately kept
**out of** `make_att_2d_masks` so that function stays bit-identical to lerobot's for
parity testing.

### And one thing lerobot gets right that looks wrong

`SmolVLAPolicy.forward` slices the loss to `action_dim`, masks, then re-slices to
`max_action_dim` and divides by `shape[-1]`. The second slice looks like a bug (it is
slicing 8 down to 32). It is a no-op, and the divisor is the real dim, which is what
you want. Test 8 confirms both reductions agree to `0.00e+00`.

---

## 7. Running it

Environment: the same conda env as the ACT/DP builds. Always prefix
`KMP_DUPLICATE_LIB_OK=TRUE` on macOS.

**Working directory matters, and the split is the same as ACT/DP.** `cfg.data_root` is
relative, so anything that touches the dataset runs from `SmolVLA/`; `rollout.py` needs
the MuJoCo scene assets and its normalisation stats come out of the checkpoint, so it
runs from `Panda/`. `parity_test.py` runs from either.

```bash
# One-time: derive the three instructions and verify the relabel is aligned.
python relabel_tasks.py --verify 9        # writes task_alignment_check.png -- LOOK at it

# Validate the whole pipeline in ~30s on CPU before spending GPU time.
python train.py --build scratch   --smoke
python train.py --build finetune  --smoke

# The real runs (100k steps, batch 16 x 4 accum = 64 effective).
python train.py --build scratch  --output_dir checkpoints/scratch
python train.py --build finetune --output_dir checkpoints/ft
python train.py --build finetune --pretrained none --output_dir checkpoints/ft_rand

# Closed-loop evaluation. Run from Panda/, not from SmolVLA/.
MUJOCO_GL=glfw python SmolVLA/rollout.py \
    --checkpoint SmolVLA/checkpoints/scratch/best.pt --episodes 200

# The language ablation: four paired conditions on byte-identical scenes.
MUJOCO_GL=glfw python SmolVLA/rollout.py \
    --checkpoint SmolVLA/checkpoints/scratch/best.pt --episodes 200 --sweep_language
```

### Training on SageMaker Studio (g6 / CUDA)

Same box and same conda env as the ACT and DP runs, so the only new step is warming the
Hugging Face cache. Run **from inside `SmolVLA/`**.

```bash
# 1. Dataset sync -- one S3 location, now three policies.
aws s3 sync s3://panda/panda_pick_place ./local_data/panda_pick_place

# 2. Verify video decode before spending GPU hours (same check as ACT/DP).
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
ds=D('panda_pick_place', root='./local_data/panda_pick_place'); \
print(ds[0]['observation.images.front'].shape)"

# 3. Warm the HF cache. THIS IS THE STEP THAT IS EASY TO SKIP.
python -c "from huggingface_hub import snapshot_download as d; \
d('HuggingFaceTB/SmolVLM2-500M-Video-Instruct'); d('lerobot/smolvla_base')"

# 4. Confirm the relabel map travelled with the code (it is committed, not derived on the box).
python -c "import json, collections as c; m=json.load(open('task_map.json')); \
print(len(m['shapes']), c.Counter(m['shapes'].values()))"    # 184 {cylinder:67, box:62, sphere:55}

# 5. The three runs. ~100M trainable params, batch 16 x 4 accum, AMP bf16 on CUDA.
python train.py --build scratch                    --data_root ./local_data/panda_pick_place --output_dir checkpoints/scratch
python train.py --build finetune                   --data_root ./local_data/panda_pick_place --output_dir checkpoints/ft
python train.py --build finetune --pretrained none  --data_root ./local_data/panda_pick_place --output_dir checkpoints/ft_rand
```

Two things that will otherwise fail late and confusingly:

- **`HF_HUB_OFFLINE=1` must not be set on the first run.** It is the right default on the
  mac (the cache is already populated and offline mode stops a stray download from
  re-resolving a moving tag), but on a fresh g6 it turns a missing snapshot into a
  `LocalEntryNotFoundError` after the dataset has already been indexed. Warm the cache in
  step 3, *then* set it if you want the guarantee.
- **Only one of the three runs actually needs `lerobot/smolvla_base`.** `scratch` and
  `finetune --pretrained none` both build their geometry from lerobot's `SmolVLAConfig`
  *dataclass*, which ships with the installed package, so they need `SmolVLM2` and nothing
  else. `smolvla_base` is downloaded only for the default `--build finetune` (its
  `model.safetensors`) and by `parity_test.py`. Step 3 fetches both anyway because the
  parity suite is the thing you want to run first on a new box.

Both `--build finetune` variants hold ~1.4 GB of fp32 frozen VLM in memory instead of the
0.7 GB lerobot would use in bf16 (see the dtype note below); that is deliberate and still
fits an L4 comfortably at batch 16.


## 8. Corrections to the paper

Recorded in full in `FLOW_MATCHING.md`; the load-bearing ones:

- **Table 10 does not show flow matching beating regression across the board.**
  Flow `89 94 85 53` (avg 80.25) vs regression `92 85 86 38` (avg 75.25): regression
  wins 2 of 4 suites, and the entire average gap is LIBERO-Long.
- **Table 8 shows 16 VLM layers is not the accuracy optimum.** 8 -> 75.0,
  16 -> 78.5, 24 -> 79.5, **32 -> 80.3**. Truncating to 16 costs ~1.8 points; it is a
  speed purchase, presented as a finding.
- **Table 9 contradicts the shipped config.** Expert width x1.00 -> 82.3 (best),
  x0.75 -> 77.5, x0.50 -> 80.3, x0.25 -> 73.8. The shipped value is x0.75, which is
  *worse than x0.50*. The prose, the table, `SmolVLAConfig` (0.75) and
  `SmolVLMWithExpertModel.__init__` (0.5) disagree three ways.
- **Tables 8/9/10 share a bit-identical anchor row** labelled both `N=32` and `x0.50`,
  while the released model is `N=16, x0.75`. Not resolvable from the paper; documented
  as unresolved rather than guessed at.
- **Section 3.1's sign convention is inconsistent with the code.** The code is
  authoritative here and is what test 7 pins down.
