# RoboTwin integration

This package contains the ModAR data converter and closed-loop evaluation
bridge for RoboTwin 2.0.

- `datagen/robotwin_to_wam.py` converts native RoboTwin HDF5 episodes into the
  intermediate ModAR trajectory layout.
- `eval/robotwin_deploy.py` adapts a checkpoint to RoboTwin observations and
  actions.
- `eval/run_sr.py` evaluates the exact held-out split reconstructed from a
  checkpoint's data configuration.

See the root [README](../README.md) and
[docs/ROBOTWIN.md](../docs/ROBOTWIN.md) for the complete workflow.
