import torch 
from torch.optim import Optimizer, AdamW, SGD   
import math

class Adam2SGD(Optimizer):
    """   
    For language modelling tasks, it is often a good idea to apply SGD for the last 
    stage of training. This class allows a transition while preserving the momentum.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        betas: coefficients for computing averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0.01)
        sgd_momentum: momentum factor for SGD (default: 0.9)
        adam_to_sgd_ratio: proportion of steps to train with AdamW (default: 0.8)
                          e.g., 0.8 means 80% AdamW, 20% SGD
        do_bias_correction: if set to False (default), initialise the moving averages to grad and grad^2 
                            rather than 0 and update without bias correction term 
    """
    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
            sgd_momentum=0.9, adam_to_sgd_ratio=0.85, do_bias_correction=False, 
            update_steps=4800, param_names=None
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= sgd_momentum:
            raise ValueError("Invalid momentum value: {}".format(sgd_momentum))
        if not 0.0 <= adam_to_sgd_ratio <= 1.0:
            raise ValueError("Invalid adam_to_sgd_ratio value: {}".format(adam_to_sgd_ratio))
        
        self.param_names = {}
        if param_names is not None:
            self.param_names = {id(p): name for name,p in param_names.items()}
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        sgd_momentum=sgd_momentum, adam_to_sgd_ratio=adam_to_sgd_ratio,
                        do_bias_correction=do_bias_correction, update_steps=update_steps)
        
        super(Adam2SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam2SGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam2SGD does not support sparse gradients')
                
                if group["weight_decay"] != 0:
                    p.data.mul_(1- group["weight_decay"] * group["lr"])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if not group["do_bias_correction"]:
                        state['exp_avg'] = grad.clone()
                        state['exp_avg_sq'] = grad.pow(2).clone()
                
                if state["step"] < group["update_steps"] * group["adam_to_sgd_ratio"]:
                    # AdamW update
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1
                    # update exponential averages
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if group["do_bias_correction"]:
                        exp_avg.div_(1 - beta1**state['step'])
                        exp_avg_sq.div_(1 - beta2**state['step'])

                    p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-group['lr']) 


                else:
                    # indicate switch to SGD
                    if state["step"] == group["update_steps"] * group["adam_to_sgd_ratio"]:
                        param_id = id(p)
                        if param_id in self.param_names:
                            print(f"Switching layer {self.param_names[param_id]} from AdamW to SGD")
                        else:
                            print(f"Switching layer {p} from AdamW to SGD")

                    # SGD update
                    state['step'] += 1
                    exp_avg = state['exp_avg']

                    # update momentum
                    exp_avg.mul_(group['sgd_momentum']).add_(grad, alpha=1 - group['sgd_momentum'])

                    # definitely totally useless either way since for large step_size, bias correction term is almost exactly 1
                    # not gonna use bias correction anyway 
                    if group["do_bias_correction"]:
                        exp_avg.div_(1 - group['sgd_momentum']**state['step'])

                    p.data.add_(exp_avg, alpha=-group['lr'])

        return loss

                    


                    

                


 


