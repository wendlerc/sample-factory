#!/usr/bin/env python3
"""
Download pretrained ViZDoom deathmatch models from HuggingFace.

Usage:
    python -m sf_examples.vizdoom.doom_arena.download_models
    python -m sf_examples.vizdoom.doom_arena.download_models --model seed0
    python -m sf_examples.vizdoom.doom_arena.download_models --train-dir ./sf_train_dir
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "seed0": {
        "repo_id": "andrewzhang505/doom_deathmatch_bots",
        "local_dir": "00_bots_128_fs2_narrow_see_0",
        "description": "Deathmatch bots seed 0 (andrewzhang505)",
    },
    "seed2222": {
        "repo_id": "edbeeching/doom_deathmatch_bots_2222",
        "local_dir": "doom_deathmatch_bots_2222",
        "description": "Deathmatch bots seed 2222 (edbeeching)",
    },
    "seed3333": {
        "repo_id": "edbeeching/doom_deathmatch_bots_3333",
        "local_dir": "doom_deathmatch_bots_3333",
        "description": "Deathmatch bots seed 3333 (edbeeching)",
    },
}


def download_model(model_key, train_dir="./sf_train_dir"):
    """Download a single model from HuggingFace."""
    info = MODELS[model_key]
    target_dir = Path(train_dir) / info["local_dir"]

    if target_dir.exists():
        print(f"  Already exists: {target_dir}")
        return target_dir

    print(f"  Downloading {info['repo_id']}...")
    downloaded = snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(target_dir),
    )
    print(f"  Saved to: {downloaded}")
    return target_dir


def main():
    ap = argparse.ArgumentParser(description="Download pretrained ViZDoom deathmatch models")
    ap.add_argument("--model", default="all", choices=["all"] + list(MODELS.keys()),
                     help="Which model to download (default: all)")
    ap.add_argument("--train-dir", default="./sf_train_dir",
                     help="Directory to store models")
    args = ap.parse_args()

    Path(args.train_dir).mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = [args.model]

    for model_key in models_to_download:
        info = MODELS[model_key]
        print(f"\n[{model_key}] {info['description']}")
        download_model(model_key, args.train_dir)

    print(f"\nAll models downloaded to {args.train_dir}")
    print("Available experiments:")
    for key, info in MODELS.items():
        target = Path(args.train_dir) / info["local_dir"]
        status = "OK" if target.exists() else "MISSING"
        print(f"  [{status}] {info['local_dir']}")


if __name__ == "__main__":
    main()
