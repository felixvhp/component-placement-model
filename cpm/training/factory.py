"""Model construction: plain name->model factory and the ablation-configured
Parallel builder."""

from cpm.models.atiss import CTATISSBoxGenerator
from cpm.models.baseline import AutoRegressiveBoxGenerator
from cpm.models.conditional_parallel import ConditionalParallelBoxGenerator
from cpm.models.parallel_transformer import ParallelBoxGenerator

MODELS = {
    "mlp": AutoRegressiveBoxGenerator,
    "ctatiss": CTATISSBoxGenerator,
    "parallel": ParallelBoxGenerator,
    "cond_parallel": ConditionalParallelBoxGenerator,
}


def make_model(name, device):
    return MODELS[name]().to(device)


def make_model_with_ablation(args, device):
    """Build a Parallel model with the requested ablation."""
    if args.model != "parallel":
        raise ValueError("Ablations only supported for --model parallel")

    kwargs = dict(
        d_model=256,
        n_layers=4,
        z_conditioning="adaln",
        vol_encoding="fourier",
        vol_concat_head=True,
    )

    if args.ablation == "baseline":
        pass
    elif args.ablation == "z_token":
        kwargs["z_conditioning"] = "token"
    elif args.ablation == "z_both":
        kwargs["z_conditioning"] = "both"
    elif args.ablation == "linear_vol":
        kwargs["vol_encoding"] = "linear"
    elif args.ablation == "no_vol_concat":
        kwargs["vol_concat_head"] = False
    elif args.ablation == "small":
        kwargs["d_model"] = 128
        kwargs["n_layers"] = 2
    elif args.ablation == "large":
        kwargs["d_model"] = 512
        kwargs["n_layers"] = 6
    else:
        raise ValueError(f"unknown ablation: {args.ablation}")

    return ParallelBoxGenerator(**kwargs).to(device)
