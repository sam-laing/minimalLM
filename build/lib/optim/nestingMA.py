from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]


import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch import Tensor
import math

ParamGroup = Dict[str, Any]

class NestedMA(Optimizer):
    r"""Implements NestedMA Optimizer Algorithm.

    just trying something out
    
    """
    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        eps: float = 1e-8,
        betas: Betas2 = (0.9, 0.999),
        do_bias_correction: bool = False,
        zero_init: bool = True,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
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
        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, 
            do_bias_correction=do_bias_correction, zero_init=zero_init)
        super().__init__(params, defaults)
    
    # not needed
    def _get_options(
        self, param_group: ParamGroup, param_shape: Tuple[int, ...]
    ) -> Tuple[bool, bool]:
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment
    
    def _get_lr(self, param_group: ParamGroup, param_state: State) -> float:
        return param_group["lr"]
        
    # not actually needed
    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    @torch.no_grad() 
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("NestedMA doesn't support sparse gradients")
                

                if group["weight_decay"] != 0:
                    #p.data = p.data + group["weight_decay"] * torch.norm(p.data, p=self.lp)
                    # decoupled weight decay as in AdamW but good ol' Lp norm easily possible too 
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if group["zero_init"]:
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["nested_exp_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        # just initialiase the exp_avg_sq as the gradient squared
                        state["exp_avg_sq"] = grad**2
                        state["nested_exp_ma"] = grad
                
                exp_avg_sq = state["exp_avg_sq"]
                nested_exp_ma = state["nested_exp_ma"]

                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Update second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["do_bias_correction"]:
                    # Compute bias-corrected second moment
                    bias_correction2 = 1 - beta2 ** state["step"]
                    corrected_exp_avg_sq = exp_avg_sq.div(bias_correction2)

                    # Update nested moving average
                    nested_exp_ma.mul_(beta1).addcdiv_(grad, corrected_exp_avg_sq.sqrt().add_(group["eps"]), value=1 - beta1)
                else:
                    nested_exp_ma.mul_(beta1).addcdiv_(grad, exp_avg_sq.sqrt().add_(group["eps"]), value=1 - beta1)

                # Compute bias-corrected first moment
                if group["do_bias_correction"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    corrected_nested_ma = nested_exp_ma.div(bias_correction1)
                else: 
                    corrected_nested_ma = nested_exp_ma
                    

                # Update parameters
                step_size = group["lr"] * corrected_nested_ma 
            
                p.data.add_(-step_size)

        return loss