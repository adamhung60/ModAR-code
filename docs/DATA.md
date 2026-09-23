# Data format

Training datasets use one directory per demonstration:

```text
<data_root>/<task>/demo_<id>/
├── trajectory.pt
└── metadata.json
```

`trajectory.pt` contains:

- `dino_tokens`: frozen DINOv2 spatial features at visual keyframes.
- `depths_png`: encoded metric-depth frames.
- `images_jpg`: encoded RGB frames.
- `point_tracks`: future patch-center tracks and visibility.
- `proprio`: dense robot configurations on the policy clock.
- `actions`: dense action targets on the policy clock.
- `visual_only`: true for actionless video packs.
- geometry, stride, trajectory-length, and validity metadata.

Visual-only packs have zero-filled `proprio` and `actions`; validity masks keep
action loss disabled. Their visual targets remain valid for dynamics training.
For robot packs, proprioception conditions every model stream through adaLN but
is never itself predicted.

## RoboTwin pipeline

The recommended paper-reproduction dataset is hosted at
[`ahung0/ModAR-RoboTwin6`](https://huggingface.co/datasets/ahung0/ModAR-RoboTwin6).
It contains exactly 250 training and 50 held-out demonstrations per task. Its
demo IDs and seed rows are remapped together so the default 300-demo split
recovers the original paper cohort.

To create an independent dataset, `scripts/data/derive_robotwin6.py` runs the
public pipeline:

1. `robotwin_manip.datagen.robotwin_to_wam` converts native HDF5 episodes.
2. `extract_dino_features.py` computes DINOv2 patch tokens.
3. `extract_point_tracks.py` computes CoTracker tracks.
4. `pack_wam_target_5mod.py` writes compact training packs.
5. `finalize_robotwin6.py` validates geometry and computes depth statistics.

The expected RoboTwin geometry is 224×168, producing a 16×12 visual-token grid,
with 14-dimensional dual-arm proprioception and actions.

## Split contract

The default release uses nested per-task pools:

- 50 action-training demonstrations.
- 10 action-validation demonstrations.
- 50 dynamics-validation demonstrations.
- 250 dynamics-training demonstrations, including all 50 action-training
  demonstrations.

The default collection therefore needs 300 episodes per task: 50 held out and
250 used for training. `split_universe: 300` fixes that shuffled base universe.
Larger collections remain supported: increasing `n_dyn_train` appends
deterministically shuffled episodes while preserving the default validation and
250-demo training subsets. Closed-loop evaluation reconstructs the checkpoint's
exact split and maps each held-out demo index to the native collection's
`seed.txt`.
