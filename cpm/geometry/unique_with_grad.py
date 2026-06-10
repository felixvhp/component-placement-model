import torch
from torch.autograd import Function


class UniqueWithGrad(Function):
    @staticmethod
    def forward(ctx, x, dim):
        u, inv = torch.unique(x, return_inverse=True, dim=dim)
        ctx.save_for_backward(inv)
        ctx.dim = dim
        return u

    @staticmethod
    def backward(ctx, grad_u):
        (inv,) = ctx.saved_tensors
        dim = ctx.dim

        # UNSURE ABOUT EXACT BEHAVIOUR OF INDEX_SELECT
        grad_x = grad_u.index_select(dim, inv)  # -> shape [128, 80, 2]

        return grad_x, None
