""" 
    Base Adam
    Hacked together by Albert Catalan
"""
import math

import torch
from torch.optim.optimizer import Optimizer

from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8 import Float8Tensor, GemmInputRole


__all__ = ("FQAdam")

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

def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):
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

class BaseAdam(Optimizer):
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
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        print("lr??", lr)
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
        super(BaseAdam, self).__init__(params, defaults)

    @staticmethod
    def _subclass_load_m(p: torch.Tensor):
        raise NotImplementedError
    
    @staticmethod
    def _subclass_load_v(p: torch.Tensor):
        raise NotImplementedError
    
    @staticmethod
    def _channel_view(x):
        return x.view(x.size(0), -1)

    @staticmethod
    def _layer_view(x):
        return x.view(1, -1)

    def step(self, closure = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        for group in self.param_groups:
            i=0
            for p in group["params"]:
                if p.grad is None:
                    continue
                i+=1
                grad = p.grad.data
                beta1, beta2 = group["betas"]
                nesterov = group["nesterov"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                # Adam
                # Albert: Load state from disk.
                exp_avg = self._subclass_load_m(state)
                exp_avg_sq = self._subclass_load_v(state)
                
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg= exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq= exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Weight decay
                wd_ratio = 1
                if group["weight_decay"] > 0:
                    p.data.mul_(
                        1 - group["lr"] * group["weight_decay"] * wd_ratio
                    )

                # Step
                p.data.add_(perturb, alpha=-step_size)
        return loss
