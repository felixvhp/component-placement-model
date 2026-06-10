"""Smoke test: the training loop runs an epoch and writes a checkpoint, for both
the projector and ramp objectives."""

from argparse import Namespace

import pytest


def _args(**overrides):
    base = dict(
        model="parallel",
        training="projector",
        epochs=1,
        lr=5e-4,
        batch_size=8,
        max_seq_length=3,
        seed=999,
        run_name="",
        ablation="baseline",
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # parallel / projector
        {"model": "cond_parallel", "training": "ramp"},  # conditional / ramp
    ],
)
def test_training_runs_and_saves(tmp_path, monkeypatch, overrides):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    from cpm.training import train

    train(_args(**overrides))

    assert list((tmp_path / "experiments").glob("*.pth")), "no checkpoint was saved"
