"""Algebraic alternative to the strict-lower row recurrence, not yet deployed."""
import torch

def reference(a):
    """The row recurrence used in the local Qwen3.5 Torch fallback."""
    out=a.clone()
    for i in range(1,a.shape[-1]):
        row=out[...,i,:i].clone()
        sub=out[...,:i,:i].clone()
        out[...,i,:i]=row+(row.unsqueeze(-1)*sub).sum(-2)
    return out+torch.eye(a.shape[-1],device=a.device,dtype=a.dtype)

def triangular(a):
    """For strictly lower A, the recurrence returns (I-A)^(-1).

    Call only on the already-masked strict-lower FP32 tensor in the reference
    kernel. This does not replace the rest of DeltaNet or change its recurrence.
    """
    if a.ndim<2 or a.shape[-2]!=a.shape[-1]:raise ValueError('Expected square matrices')
    eye=torch.eye(a.shape[-1],device=a.device,dtype=a.dtype).expand_as(a)
    return torch.linalg.solve_triangular(eye-a,eye,upper=False,unitriangular=True)
