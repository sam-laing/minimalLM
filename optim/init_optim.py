"""Intialize optimizer and scheduler."""

import torch
from .lr_schedule import WarmupCosine, WSD, WarmupConstant


def intialize_optimizer(param_groups, cfg):
  """
  Intialize an optimizer.
  NOTE: we pass weight_decay to optim, but it gets overwritten by the weight_decay in param_groups!
  """
  
  if cfg.optim == 'adamw':
    optimizer = torch.optim.AdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      eps=cfg.eps,
      weight_decay=cfg.weight_decay,
      fused=cfg.fused_optim, 
    )
  
  elif cfg.optim == 'nadamw':
    optimizer = torch.optim.NAdam(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      decoupled_weight_decay=True,
      fused=cfg.fused_optim, 
    )
  elif cfg.optim == "custom_adamw":
    from .custom_adamw import CustomAdamW
    optimizer = CustomAdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      eps=cfg.eps,
      do_bias_correction=cfg.do_bias_correction,
      zero_init=cfg.zero_init,
    )
  
  elif cfg.optim == 'sgd':
    optimizer = torch.optim.SGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )
  
  elif cfg.optim == 'signSGD':
    from .signSGD import signSGD
    optimizer = signSGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )
  
  elif cfg.optim == 'sfo_adamw':
    import schedulefree
    # warmup steps for schedulefree must be specified here
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) \
      else int(cfg.warmup_steps * cfg.steps_budget)
    optimizer = schedulefree.AdamWScheduleFree(
      param_groups,
      lr=cfg.lr,
      warmup_steps=warmup_steps,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
    )
  
  elif cfg.optim == 'nestingMA':
    """   
    - custom optimizer with adamw energy but taking moving average
    of RMSprop gradients rather than dividing two moving averages
    """
    from .nestingMA import NestedMA
    optimizer = NestedMA(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay, 
      eps=cfg.eps,
      do_bias_correction=False
    )
    
  elif cfg.optim == "muon":
    from .muon import Muon  
    optimizer = Muon(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      nesterov=cfg.nesterov,
      ns_steps=cfg.ns_steps
    )

  elif cfg.optim == "lion":
    from .lion import Lion
    optimizer = Lion(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay
    )
  elif cfg.optim == "shampoo":
    from .shampoo import Shampoo
    optimizer = Shampoo(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      weight_decay=cfg.weight_decay,
      eps=cfg.eps,
      update_freq = 1
    )
  
  elif cfg.optim == "soap":
    from .soap import SOAP
    optimizer = SOAP(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay, 
      eps=cfg.eps
    )
  elif cfg.optim == "adam2sgd":
    from .adam2sgd import Adam2SGD
    """    
    steps_budget: 4800

    # note: this is micro batch size if grad_accumulation_steps>1
    # note: with ddp, effective batch size = batch_size * grad_accumulation_steps * ddp_world_size
    micro_batch_size: 32
    grad_accumulation_steps: 8
    """
    optimizer = Adam2SGD(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      update_steps=cfg.steps_budget,
      adam_to_sgd_ratio=cfg.adam_to_sgd_ratio,
      do_bias_correction=cfg.do_bias_correction,
    )


  
  else:
    raise NotImplementedError(f"Not implemented optim: {cfg.optim}.")
  
  return optimizer


def initalize_scheduler(optimizer, cfg):
  
  if cfg.scheduler is None:
    return None
  
  # Number of warmup steps
  # either specified as a number (int) or as a percentage of steps_budget (float)
  if cfg.warmup_steps is not None and cfg.steps_budget is not None:
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else int(cfg.warmup_steps * cfg.steps_budget)
  
  # Final LR of the schedule
  # either specified as lr_end or as a percentage of lr (lr_end_pct)
  if cfg.lr_end is not None or cfg.lr_end_pct is not None:
    lr_end = cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr)

  if cfg.scheduler == "warmup_cosine":
    scheduler = WarmupCosine(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      T=cfg.steps_budget,
    )
    if cfg.optim == "adam2sgd":
      from optim.lr_schedule import WarmupCosineAdam2SGD
      scheduler = WarmupCosineAdam2SGD(
        optimizer,
        lr_start=cfg.lr_start,
        lr_max=cfg.lr,
        lr_end=lr_end,
        warmup_steps=warmup_steps,
        T=cfg.steps_budget,
      )
  elif cfg.scheduler == "wsd":
    # Number of cooldown steps
    # either specified as a number (int) or as a percentage of steps_budget (float)
    cooldown_steps = cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.cooldown_steps * cfg.steps_budget)
    cooldown_start_step = cfg.steps_budget - cooldown_steps
    scheduler = WSD(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )
    
  elif cfg.scheduler == "warmup_constant":
    scheduler = WarmupConstant(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      warmup_steps=warmup_steps,
    )
  
  else:
    raise NotImplementedError(f"Not implemented scheduler: {cfg.scheduler}.")
  
  return scheduler

