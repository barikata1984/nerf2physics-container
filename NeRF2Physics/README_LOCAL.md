# NeRF2Physics — Local Setup Notes (density-volume extension)

## Reproduce on another machine (TL;DR)

Requirements: Linux x86_64, NVIDIA GPU + driver supporting CUDA 11.8, ~40 GB
disk (pixi env 16 GB, BLIP-2 16 GB), the dataset directory
(`sledgehammer_merged_seed42/` with `complete/`, `masks/`, `transforms.json`,
`ground_truth.csv`). Inside the devcontainer of the parent repo
(`nerf2physics-container`) step 1 runs automatically as `postCreateCommand`.

```bash
cd NeRF2Physics
scripts/setup_env.sh                                        # 1. pixi + locked env + tiny-cuda-nn (~10 min)
scripts/fetch_assets.sh --dataset /path/to/sledgehammer_merged_seed42   # 2. BLIP-2 (16 GB) + scene layout
scripts/reproduce_sledgehammer.sh                           # 3. train 20k -> figures -> voxel volume -> metrics (~30 min)
```

Outputs land in `outputs/sledgehammer/reproduce/` (`comparison.png`,
`comparison_cool.png`, `density_volume.npz` on the ground-truth 128³ grid,
`metrics.json`, mayavi/slice renders). Reference outputs from the original
session are committed in `results/sledgehammer/` (incl. the LLM material
dictionary `info_new.json`, so no OpenAI/Anthropic key is needed to reproduce;
`fetch_assets.sh` copies it into the scene). NeRF training is not
bit-reproducible on CUDA — expect a few % drift in mass/COM/SWD relative to
`results/sledgehammer/metrics.json`. `ns_reconstruction.py` defaults to
`--vis_mode wandb`; the reproduce script passes `tensorboard` so no W&B key is needed.

Git layout: this directory is a `git subtree` of `ajzhai/NeRF2Physics`
(remote `nerf2physics-upstream` in the parent repo); all local changes are
listed under "What was changed vs. upstream" below.

This file documents the **local, machine-specific** setup on top of the
upstream [NeRF2Physics](https://github.com/ajzhai/NeRF2Physics) repo (branch
`local-density-volume`). It is intentionally separate from `README.md`
(upstream, unmodified) and from `arguments.py`/`predict_property.py`/etc.
(also unmodified — see "What was changed" below).

**Status legend used below:** ✅ verified on this machine · ⏳ implemented but
not yet run end-to-end (no data/API key available yet) · 🚫 blocked by an
external factor.

## What was changed vs. upstream

Additions (new files, no upstream code touched):

- `pixi.toml`, `pixi.lock` — environment definition (see below)
- `.gitignore` — added `.pixi/`, `blip2-flan-t5-xl/`, `outputs/`, `.env`
- `scripts/query_density_volume.py` — new; calls `predict_property.py`'s
  `predict_physical_property_query()` and reimplements `carving.py`'s masking
  logic (to get a boolean mask, which `carve_torch()` doesn't expose) on an
  AABB voxel grid instead of the surface points / carved grid the originals use
- `scripts/visualize_density_volume.py` — new; 3D + slice visualization of the
  resulting `density_volume.npz`
- `scripts/run_custom_scene.sh` — new; thin pipeline wrapper

One upstream file needed a small compatibility patch (`git diff ns_reconstruction.py`
shows the exact change, with inline comments explaining each hunk):

- `ns_reconstruction.py` — nerfstudio 1.1.5 (the version this pixi.toml pins,
  matching nerfstudio's own official pixi.toml) renamed/replaced two `ns-train`/
  `ns-export` CLI flags this script hard-codes:
  - `--pipeline.datamanager.camera-optimizer.mode` → `--pipeline.model.camera-optimizer.mode`
  - `--use-bounding-box`/`--bounding-box-min`/`--bounding-box-max` (axis-aligned
    corners) → `--obb-center`/`--obb-rotation`/`--obb-scale` (oriented box,
    center+rotation+scale). `obb-scale` is the box's full extent (nerfstudio's
    `OrientedBox.within()` tests against `±S/2`), so `obb-scale = bbox_size`
    reproduces the exact same axis-aligned cube centered at the origin.
  - Also added `result.check_returncode()` after each `subprocess.run(...)` call
    — the original code never checked return codes, so an `ns-train` failure
    silently fell through to a confusing `FileNotFoundError` several steps
    later instead of failing where the real error was. This is the only
    behavior change (failures now surface immediately, at the actual failing
    step); nothing changes on the success path.
- `visualization.py` — two upstream bugs/gaps, not version-drift:
  1. All four `render_pcd(...)` calls omit `hw`, so it silently defaults to
     `(1024, 1024)`. `composite_and_save()` then fails to broadcast that
     render against `orig_img` for any dataset whose images aren't 1024x1024
     (ABO-500 apparently is, ours are 800x800 -- `ValueError: operands could
     not be broadcast together with shapes (800,800,3) (1024,1024,3)`). Fixed
     by passing `hw=orig_img.shape[:2]` explicitly.
  2. `view_idx` was hardcoded to `0`. For `merged_seed42`, frame 0 happens to
     be a distant/off-center framing where the object is tiny in the image --
     the resulting comparison figure was nearly useless for judging material
     segmentation. Added `--view_idx` (`arguments.py`); when left at its
     default (-1), it now reuses the same "informative view" `captioning.py`
     already picked via `--mask_area_percentile` (stored as `idx_to_caption`
     in the info json) instead of always frame 0.
  3. `render_pcd()`'s point splats are a **fixed 8px** regardless of the
     object's actual size in the frame (`--pt_size`, hardcoded before this
     patch). For a thin object like our hammer's handle (~29px wide in the
     real photo at this framing), that inflated the rendered silhouette to
     ~40px -- a ~38% oversized "halo" beyond the true outline, made obvious
     side-by-side with `scripts/make_comparison_figure.py`. Exposed
     `--pt_size` (`arguments.py`, default 8 = unchanged upstream behavior);
     `--pt_size 3` measured out to ~31px, close to the true 29px. This is a
     rendering/display parameter only -- it does not change any predicted
     density/material value, only how tightly the point-cloud renders hug the
     true silhouette.

`scripts/make_comparison_figure.py` (new) composes `visualization.py`'s
separate RGB/Materials/Density panels into one Fig.-1-style side-by-side
image. It crops all three to **one shared** bounding box (the union of their
content) rather than cropping each independently -- otherwise a panel with
slightly different reconstructed extent gets a different crop offset, and the
"same" object part lands at different pixel positions per panel even though
`render_pcd()` renders them all from the identical camera/resolution.

Three more files got a small, additive (opt-in, default-preserving) patch to
support `--llm_provider anthropic` as an alternative to OpenAI (user request —
see section 2 "Alternative: Anthropic (Claude) instead of OpenAI"):

- `arguments.py` — added `--llm_provider {openai,anthropic}` (default `openai`)
- `gpt_inference.py` — added `claude_candidate_materials()`/`claude_thickness()`,
  which reuse the exact same system-prompt constants and few-shot examples as
  the existing `gpt_candidate_materials()`/`gpt_thickness()`, just called
  through the Anthropic Messages API instead of OpenAI's ChatCompletion
- `material_proposal.py` — dispatches to the Claude or GPT functions based on
  `--llm_provider`; the OpenAI path (default) is byte-for-byte the same
  behavior as before

## 1. Environment setup ✅

Uses [pixi](https://pixi.sh), matching
[nerfstudio's own official pixi.toml](https://github.com/nerfstudio-project/nerfstudio/blob/main/pixi.toml)
(CUDA 11.8 + PyTorch 2.2.2 + colmap 3.9.1), with NeRF2Physics's
`requirements.txt` added as pypi-dependencies. This CUDA 11.8 toolkit is
installed **inside the pixi env only** — it does not touch the container's
system CUDA (12.8) or the NVIDIA driver.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # installs to ~/.pixi/bin, no sudo
export PATH="$HOME/.pixi/bin:$PATH"            # add to your shell rc if you want this permanent

cd NeRF2Physics
pixi install            # conda (CUDA/PyTorch/colmap) + pip deps, ~16GB
pixi run tcnn-install    # builds tiny-cuda-nn from source against this env
pixi run check-cuda      # sanity check
```

Expected `check-cuda` output on this machine:
```
torch 2.2.2 cuda available: True NVIDIA GeForce RTX 3090
```

Verified working, `--help`-level: `ns-train --help`, `ns-process-data --help`,
`ns-export --help` (all exit 0), `tinycudann` import, `colmap` (3.9.1, CPU
build — same as upstream nerfstudio's own pinned pixi env), `ffmpeg`.

Verified working, **actual end-to-end run** (on the `merged_seed42` /
`sledgehammer` scene, see section 3): `ns-train nerfacto` (500-iteration debug
run), `ns-export pointcloud` (99,979 points), `ns-render` (300 depth maps) —
see section 4.

Nerfstudio's own *built-in sample datasets* (`ns-download-data blender`,
`ns-download-data nerfstudio --capture-name ...`) are still 🚫 unverified —
they're hosted on Google Drive, which rate-limited/blocked anonymous `gdown`
downloads from this machine when tried. Not needed in practice: once you have
any real scene, use that instead (see section 3).

### Known dependency conflicts this pixi.toml pins around

These are documented as comments in `pixi.toml` too:

| Symptom | Cause | Fix |
|---|---|---|
| `UserWarning: Failed to initialize NumPy` / crash risk | conda pytorch 2.2 is built against the NumPy 1.x C-API; unpinned pip installs NumPy 2.x | `numpy < 2` (both conda and pypi sections) |
| `tiny-cuda-nn` build fails: `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥ 81 dropped `pkg_resources` (upstream removal notice: slated 2025-11-30) | `setuptools < 81` (both conda and pypi sections) |
| `tiny-cuda-nn` build fails: `#error unsupported GNU version! gcc versions later than 11 are not supported!` | container's system gcc is 13; CUDA 11.8's `nvcc` requires ≤ 11 | added `gcc_linux-64`/`gxx_linux-64 = "11.*"` conda deps (isolated to the pixi env; system gcc untouched) |
| `ns-train`/`ns-export` CLI errors: `Unrecognized options: --pipeline.datamanager.camera-optimizer.mode` / no `--use-bounding-box` | nerfstudio 1.1.5 API drift vs. what `ns_reconstruction.py` was written against | patched `ns_reconstruction.py` (see "What was changed vs. upstream" above) |
| `Blip2ForConditionalGeneration requires the PyTorch library but it was not found` even though `import torch` works fine | latest `transformers` (5.x as of this session) hard-requires `torch>=2.5` and disables its whole PyTorch backend if not met; we're pinned to `torch==2.2.2` for CUDA 11.8/tinycudann compatibility | `transformers < 4.50` (pypi section) — still supports `Blip2ForConditionalGeneration`/`AutoProcessor`, doesn't gate on torch 2.5 |

## 2. API key setup ⏳ (not yet configured — no key available in this session)

The upstream code uses the **pre-1.0 OpenAI SDK** (`openai==0.28`,
`openai.ChatCompletion.create(...)`, `response['choices'][0]['message']['content']`).
Per the task's own preference, we pinned this old SDK rather than patching the
prompt/parsing code — `openai==0.28` is still installable from PyPI and is
already in `pixi.toml`.

**Important**: a ChatGPT Plus/Pro subscription (chatgpt.com) does **not**
include API credits — `platform.openai.com` API usage is billed separately
(pay-as-you-go). Same applies to a Claude Pro/Max subscription (claude.ai) vs.
the Anthropic API (console.anthropic.com) used by the `--llm_provider anthropic`
option below. Both need their own API key + billing, separate from the chat
subscription.

```bash
# Option A (what the upstream README documents):
echo "OPENAI_API_KEY = '<yourkey>'" >> my_api_key.py   # gitignored already

# Option B (also works, gpt_inference.py reads openai.api_key which
# material_proposal.py's __main__ sets from my_api_key.OPENAI_API_KEY --
# if you'd rather use an env var, set it in my_api_key.py instead:
echo "import os; OPENAI_API_KEY = os.environ['OPENAI_API_KEY']" >> my_api_key.py
export OPENAI_API_KEY=...   # set this in your own shell / .env, never commit it
```

Note: `gpt_model_name` defaults to `gpt-3.5-turbo` (`arguments.py`), which may
no longer be available on your OpenAI account by the time you run this —
override with `--gpt_model_name` if needed. This does not change prompt
semantics or output parsing.

### Alternative: Anthropic (Claude) instead of OpenAI

`material_proposal.py --llm_provider anthropic` (default remains `openai`,
fully unchanged) routes the exact same system prompts and few-shot examples
from `gpt_inference.py` through the Anthropic Messages API instead, reusing
the same `parse_material_list`/`parse_material_hardness` parsers unmodified —
prompt semantics and output format are identical, only the API call differs.
`--gpt_model_name` doubles as the Claude model name in this mode. Requires
`ANTHROPIC_API_KEY` in the environment (the Anthropic SDK reads it directly,
no key file needed):

```bash
export ANTHROPIC_API_KEY=...
pixi run python material_proposal.py --data_dir <data_dir> --start_idx 0 --end_idx 1 \
    --property_name density --llm_provider anthropic --gpt_model_name claude-haiku-4-5-20251001
```

One caveat: Anthropic's API has no `seed` parameter (OpenAI's is "best effort"
anyway), so the `gpt_wrapper()` retry-on-parse-failure loop in
`material_proposal.py` relies purely on Claude's own sampling variation across
retries rather than an explicit seed change.

## 3. Custom dataset layout ✅ (verified with `merged_seed42`)

Based on reading `ns_reconstruction.py` / `utils.py` / `arguments.py`, each
scene must look like:

```
<data_dir>/
  scenes/
    <scene_name>/
      images/               # RGBA PNGs -- see "RGBA mask" note below
      transforms.json       # nerfstudio-format camera intrinsics/extrinsics
  splits.json                # only needed if you use --split other than "all"
```

`transforms.json` is exactly nerfstudio's own format (`fl_x, fl_y, cx, cy,
frames[].transform_matrix`, camera-to-world). If your data isn't already in
this format, convert it with `ns-process-data` (images or video ->
COLMAP -> `transforms.json`), **not** by hand.

**RGBA mask assumption** (`utils.py: load_images()`): images are loaded as
4-channel PNGs and `img[:, :, 3] > 0` is used directly as the object mask.
If your custom images are plain RGB (no alpha), this will crash with an
index error on channel 3. If/when this comes up, the minimal fix is a
pre-processing step that writes an alpha channel (e.g. from a separate mask
image or a background-removal model) into the PNGs under `images/` — done
*before* handing data to NeRF2Physics, not by patching `utils.py`.

**Metric scale**: nothing in the pipeline tells you whether `transforms.json`
poses are in real-world meters. That depends entirely on how your custom
data's camera poses were produced (COLMAP alone gives an arbitrary scale;
poses from a metric-calibrated capture/AR framework are real-world). Verify
this yourself before trusting absolute mass numbers (see step 8 below).

### Worked example: `merged_seed42` (source: `/workspace/merged_seed42`)

This is a synthetic/simulated multi-view capture of one object ("sledgehammer"),
generated for a *different*, richer research pipeline (vision + robot
force/torque dynamics identification for physical-parameter estimation — the
per-frame `twist_sen`/`wrench`/`regressor` fields and the top-level
`noise_model` block in `transforms.json` are for that other pipeline and are
simply ignored by NeRF2Physics's parser). What NeRF2Physics actually needs is
a subset of it:

```
/workspace/merged_seed42/
  complete/*.png        # 300 renders, 800x800, RGBA -- alpha already == masks/*.png exactly
  masks/*.png           # redundant with complete/'s alpha channel; not needed by NeRF2Physics
  transforms.json       # nerfstudio-format top level (camera_angle_x/y, cx, cy, fl_x, fl_y, h, w)
                         # + per-frame transform_matrix; extra fields are harmless noise to our parser
  ground_truth.csv       # true 128^3 voxel mass_density field + total_mass/COM/inertia tensor --
                         # NOT consumed by the pipeline, but extremely useful for comparing our
                         # predicted density_volume.npz against ground truth (see "Important
                         # interpretation" at the top of the original task -- this dataset is exactly
                         # the kind of check that distinction calls for)
```

Converted (non-destructively, via symlinks — original data untouched) into:

```bash
SCENE_DIR=/workspace/NeRF2Physics/data/merged_seed42/scenes/sledgehammer
mkdir -p "$SCENE_DIR"
ln -sfn /workspace/merged_seed42/complete      "$SCENE_DIR/images"     # NeRF2Physics reads this
ln -sfn /workspace/merged_seed42/complete      "$SCENE_DIR/complete"   # nerfstudio's file_path="complete/..." reads this
ln -sfn /workspace/merged_seed42/masks         "$SCENE_DIR/masks"
ln -sfn /workspace/merged_seed42/ground_truth.csv "$SCENE_DIR/ground_truth.csv"
cp      /workspace/merged_seed42/transforms.json  "$SCENE_DIR/transforms.json"
```

No RGBA patch was needed here — `complete/`'s alpha channel already exactly
equals `masks/`'s content, verified pixel-for-pixel on frame 0000.

## 4. NeRF training ✅ (verified: `sledgehammer` scene, 500-iter debug run)

```bash
pixi run python ns_reconstruction.py --data_dir <data_dir> --start_idx 0 --end_idx 1 --training_iters 500
```

Wraps `ns-train nerfacto`, `ns-export pointcloud`, and `ns-render` (raw depth)
for one scene, writing everything under `<data_dir>/scenes/<scene>/ns/`
(checkpoints, `dataparser_transforms.json`, `point_cloud.ply`,
`renders/depth/`). Once this has run once, none of the later steps need to
retrain — they only read these files.

Result on `sledgehammer` with `--training_iters 500` (fast debug config, not
final quality — bump to the default 20000 for real results):
`point_cloud.ply` with 99,979 points, 300 depth maps under `ns/renders/depth/`,
`dataparser_transforms.json` with `scale ≈ 1.017`.

### IMPORTANT: upstream's hard-coded sampling flags break on small/thin objects

`ns_reconstruction.py` passes `--pipeline.model.proposal-initial-sampler uniform
--near-plane 0.4 --far-plane 6.0` (tuned for ABO-500, whose objects fill the
±1 box). On `merged_seed42` (cameras at r≈0.98, object 0.37 tall, handle
~0.02–0.05 thick, 1.6% of pixels) this under-samples the object: uniform
spacing over [0.4, 6.0] is 0.022 ≈ handle thickness and 75% of samples lie
beyond the scene. Accumulation on the object stalls at ~0.5–0.6 early on, and
with `--background-color random` the least-squares-optimal object color then
exceeds 1 (`pred + bg·(1−acc)` vs. GT≈0.89) → the RGB sigmoid saturates and
never recovers (dead gradient). Symptom: renders are solid white→yellow with
B≈0, PSNR still looks fine because the background dominates it, depth maps
are semi-transparent, and downstream carving/mass collapse to 0.

Verified by fault-tree analysis (3000-iter contrasts on the same data):
paper flags + random → saturated (acc 0.80); piecewise/near 0.05/far 2.0 +
random → correct color (acc 0.98); nerfacto defaults + random → correct;
paper flags + white bg → correct. Data, loss-side compositing, output writer,
tcnn (torch impl reproduces) and appearance embedding were excluded.

Fix applied: `--proposal_initial_sampler` added to `arguments.py` (default
`uniform`, i.e. upstream behavior) and threaded through `ns_reconstruction.py`.
For this dataset use:

```bash
pixi run python ns_reconstruction.py --data_dir data/merged_seed42 --start_idx 0 --end_idx 1 \
    --training_iters 20000 --near_plane 0.05 --far_plane 2.0 --proposal_initial_sampler piecewise
```

Paper-procedure result on `sledgehammer` (20k iters, otherwise paper
settings): 100,045-pt cloud, steel head / hickory-wood handle segmentation,
predicted mass **[0.25 – 0.67 kg]** vs GT 1.12 kg, figure at
`viz/sledgehammer_paper/sledgehammer_paper_comparison.png`. The earlier
24k-iter run with upstream flags is kept as
`data/merged_seed42/scenes/sledgehammer/ns_broken_24k_uniform/` for reference.

## 5–6. NeRF2Physics preprocessing + density inference

```bash
pixi run python feature_fusion.py   --data_dir <data_dir> --start_idx 0 --end_idx 1
pixi run python captioning.py       --data_dir <data_dir> --start_idx 0 --end_idx 1 \
    --blip2_model_dir <path to Salesforce/blip2-flan-t5-xl checkpoint>
pixi run python material_proposal.py --data_dir <data_dir> --start_idx 0 --end_idx 1 \
    --property_name density
```

✅ **`feature_fusion.py` verified** on `sledgehammer`: 766 downsampled surface
points × 300 views of CLIP (ViT-B-16, `datacomp_xl_s13b_b90k`) patch features.

✅ **`captioning.py` verified** on `sledgehammer`: caption `"a hammer"`
(reasonable, given the ground-truth object is a sledgehammer), saved to
`info_new.json`.

⏳ **`material_proposal.py` not yet run** — needs `OPENAI_API_KEY` (see
section 2), not provided in this session.

### BLIP-2 checkpoint — get only the safetensors, not the redundant `.bin` copy

The HF repo ships the *same* weights twice (`*.safetensors` **and**
`pytorch_model-*.bin`, ~15.8GB + ~15.7GB). `transformers` prefers safetensors
automatically when both exist, so the `.bin` files are pure waste — exclude
them before cloning:

```bash
git clone https://huggingface.co/Salesforce/blip2-flan-t5-xl /path/to/blip2-flan-t5-xl
cd /path/to/blip2-flan-t5-xl
git config lfs.fetchexclude "pytorch_model-00001-of-00002.bin,pytorch_model-00002-of-00002.bin"
git lfs pull   # only needed if the clone already grabbed the .bin content; harmless if not
```

(If you already cloned without excluding first, `git config lfs.fetchexclude ...`
+ `git reset --hard HEAD` + `git lfs pull` retroactively drops the `.bin`
content without re-downloading the safetensors you already have.)

Pass the directory via `--blip2_model_dir` (never hard-code the path). Verified
on this machine: `Blip2ForConditionalGeneration.from_pretrained(...)` loads
successfully onto the RTX 3090 (3.94B params).

**Known naming quirk** (`arguments.py`): `captioning.py` and
`material_proposal.py` both default to writing `info_new.json`
(`--caption_save_name` / `--mats_save_name` default to `info_new`), but
`predict_property.py`'s own default for **reading** that file
(`--mats_load_name`) is `info` (not `info_new`). If you regenerate captions/
materials yourself (rather than using the info.json shipped with the ABO-500
dataset), you must pass `--mats_load_name info_new` explicitly to
`predict_property.py` (and to `scripts/query_density_volume.py`, which
defaults to `info_new` for exactly this reason).

Standard (surface / query-point) density prediction, unmodified upstream
behavior:
```bash
pixi run python predict_property.py --data_dir <data_dir> --start_idx 0 --end_idx 1 \
    --property_name density --mats_load_name info_new --prediction_mode integral
```

## 7. Voxel-volume generation ⏳ (this session's extension)

```bash
pixi run python scripts/query_density_volume.py \
    --scene_dir <data_dir>/scenes/<scene_name> \
    --mats_load_name info_new \
    --resolution 64 64 64 \
    --output outputs/<scene_name>/density/density_volume.npz
```

- AABB defaults to the scene's own point-cloud bounding box (10% buffer,
  same helper `carving.get_bounding_box()` uses) if `--aabb_min`/`--aabb_max`
  aren't given.
- Density comes from `predict_property.py`'s `predict_physical_property_query()`,
  called **unmodified**, in batches of `--query_batch_size` (default 8192) —
  points are never all placed on the GPU at once.
- Occupancy reuses `carving.py`'s carve-by-visual-hull-and-depth-consistency
  voting logic (same algorithm as `get_carved_pts()`), applied to our own
  voxel centers. `density_final = occupancy * density_predicted`, per the
  task's occupancy/density distinction — this is an image-consistency proxy
  for "is there a surface here", not a measurement of hidden interior mass.
- `--report_mass` computes `sum(density * occupancy) * voxel_volume`, but
  prints **"metric scale unavailable; absolute mass is not valid"** unless you
  pass `--assume_metric` (only do this after verifying your poses are metric —
  see step 3).

Output `density_volume.npz` contents: `density [Nx,Ny,Nz] kg/m^3`,
`occupancy [Nx,Ny,Nz] bool`, `x_centers`/`y_centers`/`z_centers`, `aabb_min`,
`aabb_max`, `voxel_size`, `coordinate_system` (a string documenting that this
is the same frame as `utils.load_ns_point_cloud()`'s output / `transforms.json`
poses, **not** nerfstudio's internal `[-1,1]^3` training box).

## 8. Visualization ✅ (verified: `sledgehammer`)

```bash
pixi run python scripts/visualize_density_volume.py \
    --npz outputs/<scene_name>/density/density_volume.npz
```

Writes `density_3d.png` (Open3D point cloud of occupied voxels, colored by
density) and `density_slices.png` (matplotlib XY/XZ/YZ center slices) next to
the input `.npz`. Open3D off-screen rendering needs a working display/EGL
backend; if that fails in your environment, the script falls back to writing
a colored `.ply` you can open in any point-cloud viewer instead of failing.

### Alternative: mayavi volumetric rendering (same technique as pixi-wisp's `nemd_tracker`)

```bash
pixi run python scripts/visualize_density_volume_mayavi.py \
    --npz outputs/<scene_name>/density/density_volume.npz
```

`scripts/mayavi_vol_renderer.py` + `scripts/visualize_density_volume_mayavi.py`
are adapted from `wisp/trainers/tracker/mayavi_vol_renderer.py` /
`nemd_tracker.py` in the same author's
[pixi-wisp](https://github.com/barikata1984/pixi-wisp) repo (same headless
EGL→OSMesa offscreen-rendering workaround, same `get_custom_colormap`/
`init_mayavi_vol_renderer` volume-actor setup), trimmed down to a single
static render (no per-epoch tracking/video). Unlike the Open3D point cloud
above, this does true ray-cast volume rendering (`mlab.pipeline.volume`) with
unoccupied voxels made fully transparent via the opacity transfer function
(rather than plotted as discrete points), which reads noticeably cleaner for
an elongated, mostly-thin object like a hammer.

Needs `mayavi`/`pyqt`/`pyside6` (already in `pixi.toml`; conda-forge's mayavi
build pins `vtk-base < 9.5`, compatible with our existing `numpy < 2` pin).
This container's `/dev/dri/renderD*` exists but isn't accessible to the
non-root user — exactly the case `mayavi_vol_renderer.py`'s headless fix
targets — and the fix worked here on the first try.

## Full pipeline, one scene

```bash
scripts/run_custom_scene.sh <data_dir> <scene_name> 64
```

## Validation status (honest checklist, per this session)

All items below are ✅ verified end-to-end on the `sledgehammer` scene from
`merged_seed42` (500-iteration debug NeRF — not a final-quality reconstruction):

- [x] Python imports succeed (pixi env)
- [x] CUDA visible from PyTorch (`torch.cuda.is_available() == True`, RTX 3090)
- [x] Nerfstudio CLIs start (`ns-train`/`ns-process-data`/`ns-export --help`, exit 0)
- [x] `tinycudann` builds and imports
- [x] `ns_reconstruction.py`, `feature_fusion.py`, `captioning.py` all completed
- [x] density proposal obtained — **not via the OpenAI/Anthropic API** (no key
      was configured this session), but by manually running the exact same
      system prompts / few-shot examples through a Claude Opus 5 subagent and
      validating the response with the real `parse_material_list()` before
      writing it into `info_new.json` in the same shape
      `predict_candidate_materials()`/`predict_thickness()` would have. See
      "material_proposal.py without an API key" below before trusting this as
      a real substitute for automated runs.
- [x] standard (unmodified) `predict_property.py --prediction_mode integral`:
      predicted mass **[0.164 – 0.435 kg]** vs. `merged_seed42`'s ground-truth
      `total_mass` **1.121 kg** (~2.5–6.8x under). Expected given the debug NeRF,
      a generic one-word caption ("a hammer" — doesn't convey it's a *sledge*
      hammer with a heavy steel head), and the `correction_factor=0.6` heuristic.
- [x] arbitrary 3D query point density prediction (`scripts/query_density_volume.py`)
      — fixed one real bug in the process: `predict_physical_property_query()`
      also reads `args.show_mat_seg`, which the script's own minimal arg
      namespace didn't originally include (`AttributeError`, now fixed)
- [x] voxel grid saved to `.npz` — 64³ grid, 17,112/262,144 occupied (6.5%),
      raw (non-metric) mass-equivalent sum 0.38, same order of magnitude as
      `predict_property.py`'s integral prediction above (a useful
      cross-check between the two independent computation paths)
- [x] 3D density visualization generated — `density_3d.png` and
      `density_slices.png`. The hammer's silhouette (head + handle) is clearly
      recognizable in both, confirming geometry (NeRF → occupancy carving →
      voxelization) works correctly. Density values cluster narrowly
      (2157–2842 kg/m³) instead of clearly separating the steel head
      (7700–8050) from the wood/rubber/plastic handle (600–1500) — with only
      766 source feature points from a 500-iter debug NeRF and k=1 nearest-
      neighbor interpolation, material classification isn't spatially sharp
      here. Worth re-checking after a full-length (20000-iter) NeRF training.
- [x] official (patched for the render-resolution bug + view_idx default
      above) `visualization.py --scene_name sledgehammer`: produces the same
      RGB / CLIP-PCA / material-segmentation / density panel layout as the
      paper's Fig. 1/4. With a properly-framed view, material segmentation
      correctly separates the steel head (blue) from the hickory-wood handle
      (orange), and the density panel shows a visibly denser head vs. handle
      -- a materially better result than the first (badly-framed) attempt,
      though this still uses the full 99,979-point dense point cloud for
      per-pixel classification here, not the 766-point downsampled features
      the voxel-grid script (`query_density_volume.py`) uses.
- [ ] comparison against `merged_seed42`'s `ground_truth.csv`'s **per-voxel**
      `mass_density` field (not just the scalar `total_mass` used above) —
      not yet attempted, no comparison script written

### `material_proposal.py` without an API key

For this session, `info_new.json`'s `candidate_materials_density`/`thickness`
fields were produced by sending the *exact* system prompts and few-shot
examples from `gpt_inference.py` to a one-off Claude Opus 5 subagent (not
`material_proposal.py --llm_provider anthropic`, which wasn't runnable without
`ANTHROPIC_API_KEY`), and validating the reply parsed correctly with the real
`parse_material_list()` before writing it into `info_new.json` in the same
shape the script itself would produce. This is a manual, one-off stand-in for
a single test scene — not a substitute for actually running
`material_proposal.py` (via OpenAI or `--llm_provider anthropic`) once a real
API key is available, which is needed for running this automatically across
many scenes.
