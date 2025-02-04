""" 
    ADAM with INT8 CDF preserving state memory
    Hacked together by Albert Catalan
"""
import math

import torch
from torch.optim.optimizer import Optimizer
from .BaseAdam import BaseAdam


__all__ = ("CDFQAdam")

def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max

def get_quantiles(tensor: torch.Tensor, n_bits) -> torch.Tensor:
    """
        Let f:X->Y be a cdf function
        then, given certain bins of the Y, we want to find
        the x\inX that are closest.
    """
    q_min, q_max = _get_quant_range(n_bits, False)
    q_range = q_max-q_min + 1
    sorted_tensor = torch.sort(tensor).values
    cdf = torch.arange(1, len(sorted_tensor) + 1, dtype=torch.float32) / len(sorted_tensor)
    cdf_bins = torch.linspace(0, 1, steps=q_range)
    quantile_bins = torch.zeros_like(cdf_bins)
    # For every bin in cdf_bins, find the sorted_tensor value with a cdf closes to it.
    for i, bin in enumerate(cdf_bins):
        diffs =  (cdf-bin).abs()
        id = torch.argmin(diffs)
        quantile_bins[i]=sorted_tensor[id]
    return quantile_bins

def cdf_quantization(
        w: torch.tensor, n_bits
    ):
    _, q_max = _get_quant_range(n_bits, sym=False)
    qmap = get_quantiles(w, n_bits)
    differences = torch.ones_like(w)*q_max
    quantized = torch.zeros_like(w)
    dequantized = torch.zeros_like(w)
    for i in range(len(qmap)):
        qi = qmap[i]
        diff = (qi-w).abs()
        mask = differences>diff
        differences[mask]=diff[mask]
        quantized[mask] = i
        dequantized[mask] = qi
    return quantized, qmap

def cdf_dequantization(
        w_q: torch.tensor, qmap
    ):
    return torch.tensor([qmap[i] for i in w_q.to(torch.int)])

class CDFQAdam(BaseAdam):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas= (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if delta < 0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if wd_ratio < 0:
            raise ValueError("Invalid wd_ratio value: {}".format(wd_ratio))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super().__init__(params, **defaults)

    @staticmethod
    def _subclass_load_m(state):
        quant, qmap = cdf_quantization(state["exp_avg"], 8)
        dequant = cdf_dequantization(quant, qmap)
        return dequant

    @staticmethod
    def _subclass_load_v(state):
        quant, qmap = cdf_quantization(state["exp_avg_sq"], 8)
        dequant = cdf_dequantization(quant, qmap)
        return dequant
