"""CLI entry point for training. The implementation lives in :mod:`cpm.training`."""

import argparse

from cpm.training import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["mlp", "ctatiss", "parallel", "cond_parallel"],
        default="ctatiss",
    )
    p.add_argument("--training", choices=["ramp", "projector"], default="ramp")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument(
        "--ablation",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "z_token",
            "z_both",
            "linear_vol",
            "no_vol_concat",
            "small",
            "large",
        ],
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
