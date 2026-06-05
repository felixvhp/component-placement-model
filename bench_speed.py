"""
Inference / backprop speed comparison: AR-Transformer vs Parallel decoder.

For each (model, seq_len) we time:
  - forward only (inference latency)
  - forward + backward (the GLUE-relevant gradient path)
averaged over many repetitions, repeated across several random seeds so we can
report mean +/- std. Uses randomly-initialized models (timing depends only on
architecture and shapes, not on trained weights) and CUDA synchronization with
warmup so the numbers reflect real device time.

Outputs a JSON with per-(model, seq_len, mode) mean/std in milliseconds.
"""

import argparse
import json
import time

import torch

from src.config import latent_dim
from src.geometry.internals import BatchInternals
from src.models.atiss import CTATISSBoxGenerator
from src.models.parallel_transformer import ParallelBoxGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "ctatiss": CTATISSBoxGenerator,
    "parallel": ParallelBoxGenerator,
}


def sample_inputs(batch, seq_len):
    z = torch.randn(batch, latent_dim, device=device)
    vols = 0.005 + torch.rand(batch, seq_len, device=device) * (0.1 - 0.005)
    return vols, z


def scalar_loss(boxes):
    """A cheap scalar that depends on all box outputs, so backward exercises the
    full graph (mirrors the overlap objective used in training/eval)."""
    lwh, xyz = boxes[:, :, :3], boxes[:, :, 3:]
    internals = BatchInternals(xyz, lwh)
    return internals.checkOverlap().sum()


def time_block(model, vols, z, *, backward, reps):
    """Return per-iteration wall time in ms (device-synced)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        if backward:
            model.zero_grad(set_to_none=True)
            z_in = z.detach().requires_grad_(True)
            boxes, _ = model(vols, z_in)
            loss = scalar_loss(boxes)
            loss.backward()
        else:
            with torch.no_grad():
                model(vols, z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100,
                    help="batch size (matches eval: 100 latent samples)")
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50, help="timed reps per measurement")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/speed_benchmark.json")
    args = ap.parse_args()

    print(f"device={device}, batch={args.batch}, reps={args.reps}, "
          f"warmup={args.warmup}, seeds={args.seeds}")

    results = {}
    for mname, Mclass in MODELS.items():
        for sl in args.seq_lens:
            for mode in ("forward", "fwd_bwd"):
                backward = mode == "fwd_bwd"
                per_seed = []
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    model = Mclass().to(device)
                    if not backward:
                        model.eval()
                    vols, z = sample_inputs(args.batch, sl)
                    # warmup (compiles kernels, fills caches)
                    time_block(model, vols, z, backward=backward, reps=args.warmup)
                    ms = time_block(model, vols, z, backward=backward, reps=args.reps)
                    per_seed.append(ms)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                t = torch.tensor(per_seed)
                key = f"{mname}_sl{sl}_{mode}"
                results[key] = {
                    "mean_ms": t.mean().item(),
                    "std_ms": t.std(unbiased=False).item(),
                    "per_seed_ms": per_seed,
                }
                print(f"{key:28s} {t.mean():8.3f} +/- {t.std(unbiased=False):6.3f} ms")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": str(device), "batch": args.batch,
                   "reps": args.reps, "results": results}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
