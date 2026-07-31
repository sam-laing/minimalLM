from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import torch
from torch.optim.optimizer import Optimizer

Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
LossClosure = Callable[[], float]
Betas2 = Tuple[float, float]
ParamGroup = Dict[str, Any]

class TrackingAdamW(Optimizer):
    """
    A modified version of AdamW optimizer with an option
    to disable bias correction and initialize moments directly with gradients.


    - zero_init: bool, if True exp_avg and exp_avg_sq are initialized as zero and if False with g1, g1^2
    - do_bias_correction: bool, if True bias correction is applied (i.e \hat{m_t} = mt / (1-beta1^t)), else not applied (\hat{m}_t = m_t)
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        eps: float = 1e-8,
        betas: Betas2 = (0.9, 0.999),
        do_bias_correction: bool = False,
        zero_init: bool = False,
        weight_decay: float = 0.0,
        eps_inside_sqrt: bool = False, 
        model: Optional[torch.nn.Module] = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            do_bias_correction=do_bias_correction,
            zero_init=zero_init,
            eps_inside_sqrt=eps_inside_sqrt,
        )
        super().__init__(params, defaults)

        if model is not None:
            self.model = model
            # make an id to name dict
            self.id_to_name = {id(p): name for name, p in model.named_parameters()}

    @torch.no_grad()
    def step(
        self, closure: LossClosure = None, 
        log_gradients: bool = False, 
        log_moments: bool = False
        ) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        if log_gradients:
            self.grad_dict = {}
        if log_moments:
            self.moment_dict = {}
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                
                if log_gradients:
                    if self.model is not None:
                        name = self.id_to_name.get(id(p), str(id(p)))
                        self.grad_dict[name] = grad.clone().detach()
                    else:
                        self.grad_dict[id(p)] = grad.clone().detach()

                #apply weight decay (decoupled as in AdamW)
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                state = self.state[p]
                #init state
                if len(state) == 0:
                    state["step"] = 0
                    if group["zero_init"]:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"] = grad.clone()
                        state["exp_avg_sq"] = grad.pow(2).clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                #update ema with current gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #optional bias correction addition
                if group["do_bias_correction"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    exp_avg_corrected = exp_avg / bias_correction1
                    exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                else:
                    exp_avg_corrected = exp_avg
                    exp_avg_sq_corrected = exp_avg_sq

                if log_moments:
                    if self.model is not None:
                        name = self.id_to_name.get(id(p), str(id(p)))
                        self.moment_dict[name] = {
                            "exp_avg": exp_avg_corrected.clone().detach(),
                            "exp_avg_sq": exp_avg_sq_corrected.clone().detach(),
                        }
                    else:
                        self.moment_dict[id(p)] = {
                            "exp_avg": exp_avg_corrected.clone().detach(),
                            "exp_avg_sq": exp_avg_sq_corrected.clone().detach(),
                        }

                #finally update the parameters
                if group["eps_inside_sqrt"]:
                    # add eps to exp_avg_sq before taking the square root
                    denom = exp_avg_sq.add_(group["eps"]).sqrt()
                else:
                    denom = exp_avg_sq_corrected.sqrt().add_(group["eps"])

                p.data.addcdiv_(exp_avg_corrected, denom, value=-group["lr"])

        return loss
    
    def get_grad_dict(self) -> dict:
        """
        Returns the dictionary of gradients if log_gradients was set to True.
        """
        if hasattr(self, 'grad_dict'):
            return self.grad_dict
        else:
            raise ValueError("Gradients were not logged. Set log_gradients=True in step() method.")
    def get_moment_dict(self) -> dict:
        """
        Returns the dictionary of moments if log_moments was set to True.
        """
        if hasattr(self, 'moment_dict'):
            return self.moment_dict
        else:
            raise ValueError("Moments were not logged. Set log_moments=True in step() method.")
    