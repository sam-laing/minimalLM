""" 
    ADAM with INT8 CDF preserving state memory
    Hacked together by Albert Catalan
"""
import math

import torch
from torch.optim.optimizer import Optimizer
from .BaseAdam import BaseAdam


from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8 import Float8Tensor, GemmInputRole


__all__ = ("FP8Adam")


FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}

def fake_quant_to_fp8(x: torch.Tensor, float8_dtype: torch.dtype, verbose:bool = False):
    if torch.max(torch.abs(x)) == 0:
        scale = 1
    else:
        scale = tensor_to_scale(x, float8_dtype)
    if verbose:
        print(f"scale: {scale}")
    return hp_tensor_and_scale_to_float8(x, scale, float8_dtype).to_original_precision()

def hp_tensor_and_scale_to_float8(tensor, scale, float8_dtype, linear_mm_config=None, gemm_input_role=GemmInputRole.INPUT):
    tensor_scaled = tensor.to(torch.float32) * scale
    bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
    return Float8Tensor(
        bits_fp8,
        scale,
        tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
        )

def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype=torch.float8_e4m3fn):
    """Converts a tensor to a saturated fp8 tensor.

    Note:
        The default behavior in PyTorch for casting to `float8_e4m3fn`
        and `e5m2` is to not saturate. In this context, we should saturate.
        A common case where we want to saturate is when the history of a
        tensor has a maximum value of `amax1`, and the current amax value
        is `amax2`, where `amax1 < amax2`. This is common when using delayed
        scaling.
        https://github.com/pytorch/ao/blob/main/torchao/float8/float8_utils.py
    """
    if float8_dtype in FP8_TYPES:
        max_value = torch.finfo(float8_dtype).max
        x = x.clamp(min=-max_value, max=max_value)
        return x.to(float8_dtype)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

class FP8Adam(BaseAdam):
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
        return fake_quant_to_fp8(state["exp_avg"], torch.float8_e4m3fn)

    @staticmethod
    def _subclass_load_v(state):
        return fake_quant_to_fp8(state["exp_avg_sq"], torch.float8_e4m3fn)