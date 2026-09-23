# Training on real data

ModAR supports arbitrary mixtures through `data.sources`. Each source specifies
its own root, tasks, sampling weight, normalization statistics, and supervision
stream.

| Stream | Future modalities | Actions | Typical source |
| --- | --- | --- | --- |
| `dynamics` | yes | no | human or internet video |
| `action` | no | yes | robot demonstrations used as policy data only |
| `joint` | yes | yes | robot demonstrations used for both objectives |

The example in `conf/examples/real_data.yaml` combines an actionless video
source with a joint robot source. This is the key difference from the paper's
RoboTwin setup: RoboTwin creates separate action and dynamics streams from one
robot dataset, while a real-data mixture can deliberately use robot examples as
`joint` so they train dynamics prediction too.

## Visual-only packs

Set `visual_only: true` in `trajectory.pt` for examples without robot state or
actions. The loader emits zero proprioception/actions and invalid action masks.
The model still trains all requested visual targets. Proprioception is global
conditioning only; there is no proprioception prediction loss.

## Preparing recorded data

Convert recordings to the schema in `docs/DATA.md`. Visual-only and robot packs
use the same visual fields; robot packs additionally carry aligned qpos/actions
on the policy clock and set `visual_only: false`.

Before training:

1. Replace the example task names and source roots.
2. Compute each source's depth mean/std and update the YAML.
3. Set robot data to `stream: joint` if it should supervise dynamics.
4. Match `model.action_dim`, `model.proprio_dim`, geometry, and horizons to the
   packed data.

Then invoke the trainer with the example config:

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train/train_modality_forcing.py \
  config=conf/examples/real_data.yaml
```
