""" 
    TODO
       ADAM with int8 state memory preserving the normal distribution
    Hacked together by Albert Catalan
"""
import math

import torch
from torch.optim.optimizer import Optimizer

from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8 import Float8Tensor, GemmInputRole

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("FQAdam")

class NQAdam(Optimizer):
    r"""Implements QAdam algorithm.

    It has been proposed in __

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        delta: threhold that determines whether a set of parameters is scale
            invariant or not (default: 0.1)
        float8_dtype: bits to quantize to (default: torch.float8_e4m3fn)
        wd_ratio: relative weight decay applied on scale-invariant parameters
            compared to that applied on scale-variant parameters (default: 0.1)
        nesterov: enables Nesterov momentum (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.QAdam(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

     __ ?

    Note:
        Reference code: 
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
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
        super(NQAdam, self).__init__(params, defaults)


    @staticmethod
    def _subclass_load_m(state):
        return fake_quant_to_fp8(state["exp_avg"], torch.float8_e4m3fn)

    @staticmethod
    def _subclass_load_v(state):
        return fake_quant_to_fp8(state["exp_avg_sq"], torch.float8_e4m3fn)