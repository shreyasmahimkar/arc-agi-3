import torch

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithmic transformation.
    Compresses large magnitudes while remaining linear near zero.
    Used in DreamerV3 for rewards, values, and observations.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the symmetric logarithmic transformation.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
