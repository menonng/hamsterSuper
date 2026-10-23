HamsterSuper model-iter005000
## HamsterSuper autonomous-driving policy `model-iter005000`

- PPO iterations: 5000 / episodes: 51936
- Curriculum level: 2
- Random courses at curriculum level 2 (domain randomization ON): success rate **46.7%**, collisions per episode 14.20, mean finish time 67.1s
- Race-size courses (10-16 cells per side, round trip): success rate **10.0%**, mean finish time 15.2s
- Example maps: success rate **63.3%**

### Files
- `policy_ts.pt`: TorchScript model for the real robot/server (`python -m hamster_super.server --model policy_ts.pt`)
- `policy.npz`: the same policy for NumPy-only inference (no torch, e.g. Raspberry Pi)
- `checkpoint.pt`: resume training (`--resume`)
- `metrics.json`, `env_config.json`
