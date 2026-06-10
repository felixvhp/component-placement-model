# Checkpoints

One canonical trained checkpoint per model, committed so the repo runs out of the
box (evaluation, the interactive viewer, and GLUE integration). Other runs live in
the gitignored `experiments/` folder.

## Files

- **`mlp.pth`**: MLP baseline (`mlp`), projector training, seed 42 — GLUE's original internals model
- **`ctatiss.pth`**: ATISS-style AR transformer (`ctatiss`), projector training, seed 42
- **`parallel.pth`**: Parallel decoder (`parallel`), projector training, seed 1
- **`cond_parallel.pth`**: Conditional parallel model (`cond_parallel`), seed 42 — trained with constraint density 0.3 and target_strength 0.8

## Loading

```python
import torch
from cpm.training import make_model

model = make_model("parallel", torch.device("cpu"))
model.load_state_dict(torch.load("checkpoints/parallel.pth", map_location="cpu"))
model.eval()
```

Or evaluate directly: `python scripts/eval.py --model parallel --weights checkpoints/parallel.pth`.
