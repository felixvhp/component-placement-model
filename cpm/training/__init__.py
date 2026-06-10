"""Training: model factory, per-batch loss orchestration, and the training loop."""

from cpm.training.factory import make_model, make_model_with_ablation
from cpm.training.loop import train
from cpm.training.objective import compute_losses

__all__ = ["compute_losses", "make_model", "make_model_with_ablation", "train"]
