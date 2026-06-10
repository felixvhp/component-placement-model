"""Constraint domain: target sampling & satisfaction metrics (``sampling``)
and Levenberg-Marquardt projection of layouts onto constraints (``projection``).

Public names are re-exported here so callers can use ``from cpm.constraints
import sample_target`` without depending on the internal module split.
"""

from cpm.constraints.projection import (
    gated_ema_update,
    lm_project_step,
    make_constraint_fn,
)
from cpm.constraints.sampling import (
    constraint_loss,
    sample_target,
    satisfaction_rate,
)

__all__ = [
    "constraint_loss",
    "gated_ema_update",
    "lm_project_step",
    "make_constraint_fn",
    "sample_target",
    "satisfaction_rate",
]
