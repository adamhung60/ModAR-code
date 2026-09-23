# ModAR

This repository contains the official implementation of
**Modality-Autoregressive World-Action Models (ModAR)**. ModAR autoregressively
generates future modalities (point tracks, DINO features, depth, and RGB), then robot
actions.

The code supports the six-task RoboTwin benchmark used in the paper, together
with a generic mixed-source training interface for real-world setups.

## Supported WAM formulations

| Public name | Config | Generation formulation |
| --- | --- | --- |
| ModAR | `modar` | Fixed-order modality-autoregressive generation |
| Unified | `unified` | Joint generation with one shared noise level |
| Disjoint | `disjoint` | Independent targets with no future cross-conditioning |
| Independent-noise | `independent_noise` | Independent noise level per modality during training, joint denoising at inference |
| Action-only | `action_only` | Behavior cloning without future prediction |

## 1. Installation

```bash
git clone https://github.com/adamhung60/ModAR-code.git
cd ModAR-code
conda env create -f environment.yml
conda activate modar
```

The pinned CUDA 12.8 PyTorch build supports NVIDIA architectures from V100
(`sm_70`) through RTX 5090/Blackwell (`sm_120`).

DINOv2 weights are downloaded through `torch.hub` on first use. CoTracker is
installed from its pinned upstream commit; its code and weights are licensed
separately under CC BY-NC 4.0.

Download the CoTracker3 offline checkpoint before generating data:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth \
  -o ~/.cache/torch/hub/checkpoints/scaled_offline.pth
```

Closed-loop evaluation additionally requires a working
[RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin) checkout and its
own conda environment.

## 2. Get the RoboTwin data

We use the tasks:

```text
dump_bin_bigbin
pick_diverse_bottles
place_bread_skillet
put_bottles_dustbin
stack_bowls_three
turn_switch
```

For the paper reproduction, download
[`ahung0/ModAR-RoboTwin6`](https://huggingface.co/datasets/ahung0/ModAR-RoboTwin6),
the compact dataset containing 250 training and 50 held-out
demonstrations per task:

```bash
hf download ahung0/ModAR-RoboTwin6 \
  --repo-type dataset \
  --local-dir data/ModAR-RoboTwin6

(
  cd data/ModAR-RoboTwin6
  sha256sum -c archives/SHA256SUMS
)

for archive in data/ModAR-RoboTwin6/archives/*.tar; do
  tar -xf "$archive" -C data/ModAR-RoboTwin6
done

export MODAR_DATA_ROOT="$PWD/data/ModAR-RoboTwin6/robotwin6_packed"
export ROBOTWIN_DATA_ROOT="$PWD/data/ModAR-RoboTwin6/robotwin6_seeds"

cp robotwin_manip/datagen/configs/modar_robotwin6.yml \
  /path/to/RoboTwin/task_config/
```

The demos and seeds are consistently remapped so the public default split
recovers the paper's original training, validation, and action-labeled subsets.
No split override is required. The dataset is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); the repository
code remains MIT-licensed.

### Generate a new dataset instead

Copy `robotwin_manip/datagen/configs/modar_robotwin6.yml` into RoboTwin's
`task_config/` directory, set its `save_path`, and collect each task with
RoboTwin's standard collector. For example:

```bash
cd /path/to/RoboTwin
bash collect_data.sh dump_bin_bigbin modar_robotwin6 0
```

After collecting all six tasks, create training packs:

```bash
cd /path/to/ModAR-code
conda activate modar
export MODAR_BUILD_ROOT=/path/to/modar_data

python scripts/data/derive_robotwin6.py \
  --gpus 0 \
  --raw-root "$MODAR_BUILD_ROOT/robotwin6_raw" \
  --wam-root "$MODAR_BUILD_ROOT/robotwin6_wam" \
  --tracks-root "$MODAR_BUILD_ROOT/robotwin6_tracks" \
  --packed-root "$MODAR_BUILD_ROOT/robotwin6_packed" \
  --task-config modar_robotwin6 \
  --expected-per-task 300

python scripts/data/finalize_robotwin6.py \
  --pack-root "$MODAR_BUILD_ROOT/robotwin6_packed" \
  --expected-per-task 300 \
  --update-configs

export MODAR_DATA_ROOT="$MODAR_BUILD_ROOT/robotwin6_packed"
export ROBOTWIN_DATA_ROOT=/path/to/RoboTwin/data
```

The derivation is resumable. It converts native HDF5 episodes, extracts DINO
tokens and CoTracker tracks, packs each demonstration, and validates the result.
`--update-configs` explicitly writes the newly measured depth statistics into
`conf/methods/common.yaml`; omit it for validation without changing the checkout.
The default 300 episodes per task provide 250 nested training demonstrations
(50 of them action-labeled) and 50 held-out demonstrations. Collect more and
raise `data.n_dyn_train` to run larger-data regimes.
See [docs/DATA.md](docs/DATA.md) for the pack contract.

## 3. Train

```bash
conda activate modar
export MODAR_DATA_ROOT=/path/to/modar_data/robotwin6_packed

# One GPU
scripts/mf train modar --gpus 0

# Four GPUs
scripts/mf train modar --gpus 0,1,2,3
```

All YAML fields can be overridden from the command line:

```bash
scripts/mf train unified --gpus 0,1 \
  data.n_dyn_train=50 \
  train.batch_size=8 \
  train.max_samples=2400000 \
  train.wandb=true
```

For example, to train a custom four-modality ModAR model, change the modality
set and order together:

```bash
scripts/mf train modar --gpus 0 \
  'model.modalities=[dino,tracks,depth,action]' \
  'model.generation_order=[tracks,dino,depth,action]' \
  'model.history_modalities=[dino,depth]'
```

Inspect a fully resolved method config with `scripts/mf show modar`. Outputs are
written below `outputs/` unless `train.save_dir` is overridden.

## 4. Evaluate in RoboTwin

Run the paper's held-out-seed protocol:

```bash
conda activate RoboTwin
python robotwin_manip/eval/run_sr.py \
  --ckpt outputs/modar/last.pt \
  --data-root "$MODAR_DATA_ROOT" \
  --robotwin-data "$ROBOTWIN_DATA_ROOT" \
  --robotwin-repo /path/to/RoboTwin \
  --task-config modar_robotwin6 \
  --n-per-task 50 \
  --out outputs/modar/success_rate.jsonl
```

Evaluation reconstructs the training split and maps held-out `demo_<i>` entries
to RoboTwin's `seed.txt`. Always use the packed and seed roots from the same
download or collection. The task configuration copied in Section 2 is required
in RoboTwin's `task_config/` directory.

The direct command above does not require a policy shim. To use RoboTwin's own
`script/eval_policy.py` harness instead, install the optional shim:

```bash
conda activate RoboTwin
python scripts/data/prepare_robotwin.py --robotwin-repo /path/to/RoboTwin
```

The paper's corresponding ModAR run achieved 226/300 successes (75.3%):
94%, 60%, 74%, 82%, 68%, and 74% on the tasks in the order listed above. The
hosted dataset contains that run's exact episode cohort. A newly collected
dataset uses different training and held-out episodes, so exact episode-level
results are not expected even when the protocol is reproduced correctly.

## Training on real data

`conf/examples/real_data.yaml` demonstrates the generic source interface:

- `stream: dynamics` trains future prediction only. Visual-only packs provide
  zero proprioception and no valid action targets.
- `stream: action` trains actions only.
- `stream: joint` trains both future prediction and actions. Use this for
  action-labeled robot demonstrations when robot data should also train the
  dynamics model.

Proprioception is conditioning, never a prediction target. See
[docs/REAL_DATA.md](docs/REAL_DATA.md) for the schema and the difference from
the paper's RoboTwin split.

## Repository layout

```text
conf/methods/             published method configurations
conf/examples/            generic real-data example
scripts/data/             RoboTwin and real-data pack builders
scripts/train/            training entry point
robotwin_manip/           RoboTwin conversion and closed-loop evaluation
util/modality_forcing/    model, schedulers, data loaders, and inference
tests/                    focused model/data/deployment tests
```

## License and citation

ModAR is released under the [MIT License](LICENSE). Third-party dependencies
retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).

Paper: [Modality-Autoregressive World-Action Models](https://arxiv.org/abs/2609.17524)

Project page: https://adamhung60.github.io/ModAR/

```bibtex
@misc{hung2026modar,
  title={Modality-Autoregressive World-Action Models},
  author={Adam Hung and Bardienus P. Duisterhof and Deva Ramanan and Jeffrey Ichnowski},
  year={2026},
  eprint={2609.17524},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.17524},
}
```
