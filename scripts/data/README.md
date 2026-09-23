# Data tools

- `derive_robotwin6.py`: resumable native-HDF5 to training-pack pipeline.
- `finalize_robotwin6.py`: validate packs and compute depth statistics.
- `extract_dino_features.py`: frozen DINOv2 feature extraction.
- `extract_point_tracks.py`: patch-center CoTracker extraction.
- `pack_wam_target_5mod.py`: compact multimodal pack writer.
- `prepare_robotwin.py`: install the optional RoboTwin policy shim.

Run tools from the repository root in the `modar` conda environment. Use
`--help` for explicit input and output arguments.
