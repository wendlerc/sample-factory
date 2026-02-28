# Doom Arena

ViZDoom deathmatch training and evaluation using Sample Factory (APPO).

## Quick Start

```bash
# Install
pip install -e ".[vizdoom]"

# Download pretrained models
python -m sf_examples.vizdoom.doom_arena.download_models

# Run a pretrained agent (generates video)
python -m sf_examples.vizdoom.doom_arena.run_agent --episodes 5 --output results/showcase.mp4

# Train from scratch
python -m sf_examples.vizdoom.doom_arena.train_deathmatch \
    --experiment=my_run --with_wandb=True --train_for_env_steps=100000000
```

## Scripts

| Script | Description |
|--------|-------------|
| `download_models.py` | Download pretrained HuggingFace models |
| `run_agent.py` | Run agent and record video/frames |
| `train_deathmatch.py` | Train from scratch with APPO + wandb |
| `eval_detailed.py` | Detailed evaluation (kills, deaths, damage) |
| `monitor_training.py` | Periodic checkpoint eval + wandb video logging |
| `sample_frames.py` | Sample frames and create comparison grids |
| `log_results_wandb.py` | Log evaluation results to wandb |

## Pretrained Models

| Model | Source | Avg Reward |
|-------|--------|------------|
| seed0 | andrewzhang505/doom_deathmatch_bots | ~25 |
| seed2222 | edbeeching/doom_deathmatch_bots_2222 | ~24 |
| seed3333 | edbeeching/doom_deathmatch_bots_3333 | ~25 |

Training from scratch reaches ~30 reward within 150M frames (~1.5h on A6000).

## Environment

- **Map**: dwango5.wad (deathmatch)
- **Bots**: 7 built-in bots
- **Observation**: 128x72 RGB
- **Actions**: 39 discrete (movement + shooting + weapon switching)
- **Architecture**: ConvNet Simple -> 512 MLP -> LSTM 512
- **Algorithm**: APPO (Asynchronous PPO)

## Compatibility Patches

This fork includes fixes for:
- numpy 2.x scalar assignment
- PyTorch 2.6+ `weights_only` default
- gymnasium `env.seed()` removal
- Flexible checkpoint loading (`strict=False`)
