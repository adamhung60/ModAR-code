# RoboTwin setup

ModAR expects RoboTwin 2.0 native episodes with head-camera RGB, metric depth,
point clouds, dual-arm qpos/action vectors, and `seed.txt`.

## Collection

Copy `robotwin_manip/datagen/configs/modar_robotwin6.yml` into RoboTwin's
`task_config/` directory and change `save_path` to an absolute output path.
Collect the six tasks listed in the root README with RoboTwin's standard
`collect_data.sh` command. Keep the generated task/config directory layout;
`derive_robotwin6.py` reads it directly.

## Training packs

Run the derivation script in the `modar` conda environment. It is resumable and
accepts multiple GPU IDs. All roots are explicit, so raw data and intermediate
features can live on different volumes.

## Closed-loop evaluation

Evaluation runs in RoboTwin's environment, not the training environment. The
driver imports ModAR from this checkout and uses RoboTwin for SAPIEN task
construction. `scripts/data/prepare_robotwin.py` installs the optional policy
shim; `robotwin_manip/eval/run_sr.py` runs the held-out protocol directly.

The native archive and packed dataset must derive from the same collection.
Evaluation uses `demo_<i>` to index the corresponding native `seed.txt`; a pack
copied from another collection does not define the same initial conditions.
