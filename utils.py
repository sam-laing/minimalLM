
import os
import yaml
import math
import shutil
import wandb
import torch
import time

from itertools import product
from collections import namedtuple

def print_master(msg):
  """Prints only in master process if using multiple GPUs."""
  rank = os.environ.get('RANK', -1)
  ddp = int(rank) != -1
  master_process = (not ddp) or (int(rank) == 0)
  
  if master_process:
    print(msg)

# Add to utils.py
def get_paired_betas(cfg, job_idx=None):
    """Extract beta pairs from config based on job index"""
    if hasattr(cfg, 'beta_pairs'):
        if job_idx is None:
            return cfg.beta_pairs[0]  #default to first pair
        pair_idx = job_idx % len(cfg.beta_pairs)
        return cfg.beta_pairs[pair_idx]
    else:
        #fallback to original betas
        return cfg.beta1, cfg.beta2


def load_config(path, job_idx=None):
  """
  Parse a yaml file and return the correspondent config as a namedtuple.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """
  
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  Config = namedtuple('Config', config_dict.keys())

  if job_idx is None:
    cfg = config_dict
    sweep_size = 1

  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    sweep_size = len(combinations)
    if job_idx >= sweep_size:
      raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

    combination = combinations[job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}

  cfg_obj = Config(**cfg)


  
  return cfg_obj, sweep_size


def init_wandb(cfg):
  """Initalizes a wandb run"""
  #os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
  os.environ["WANDB__SERVICE_WAIT"] = "600"
  os.environ["WANDB_SILENT"] = "true"
  #wandb_run_name = f"{cfg.optim}, {cfg.scheduler}, lr={cfg.lr}, wd={cfg.weight_decay}, b1={cfg.beta1}, b2={cfg.beta2}, seed={cfg.seed}"

  wandb_run_name = f"{cfg.optim}, {cfg.scheduler}, eps = {cfg.eps}, lr={cfg.lr}, wd={cfg.weight_decay}, b1={cfg.beta1}, b2={cfg.beta2}, seed={cfg.seed}"
  if cfg.optim == "custom_adamw":
    wandb_run_name += f", bias_c={cfg.do_bias_correction}, zero_init={cfg.zero_init}"
  if cfg.optim == "adam2sgd":
    wandb_run_name += f", a2s={cfg.adam_to_sgd_ratio}"
  wandb.init(
    project=cfg.wandb_project, 
    name=wandb_run_name, 
    dir=cfg.wandb_dir,
    config=cfg._asdict(), 
    tags = [
      f"optim={cfg.optim}"
      f"lr={cfg.lr}",
      f"wd={cfg.weight_decay}",
      f"b1={cfg.beta1}",
      f"b2={cfg.beta2}",
      f"grad_clip={cfg.grad_clip}",
      f"eps={cfg.eps}",
      f"bs={cfg.micro_batch_size * cfg.grad_accumulation_steps}",
      f"seq_len={cfg.seq_len}",
      f"steps_budget={cfg.steps_budget}",
      f"micro_batch_size={cfg.micro_batch_size}",
      f"grad_accumulation_steps={cfg.grad_accumulation_steps}",
      f"scheduler={cfg.scheduler}",
      f"warmup_steps={cfg.warmup_steps}",
      f"do_bias_correction={cfg.do_bias_correction}",
      f"zero_init={cfg.zero_init}",
      f"seed={cfg.seed}", 
      f"lr_end={cfg.lr_end}",
      f"lr_start={cfg.lr_start}",
      f"warmup_steps={cfg.warmup_steps}",
      f"d_model={cfg.d_model}",
      f"n_layers={cfg.n_layers}",
      f"n_heads={cfg.n_heads}",
      f"expand={cfg.expand}",

    ]
  )



def maybe_make_dir(cfg, job_idx=None):
  """Creates an experiment directory if checkpointing is enabled"""
  if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
    return
  if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
    return
  
  # Use cfg.exp_name instead of generating a dynamic name
  exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
  if job_idx is not None:  # subfolder for each job in the sweep
    exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")

  if os.path.exists(exp_dir):
    if not cfg.over_write:
      raise ValueError(f"Experiment dir: {exp_dir} already exists and over_write=False")
    print(f"Removing experiment dir: {exp_dir}")
    shutil.rmtree(exp_dir)  # Add import for shutil at the top of the file

  print(f"Creating experiment directory: {exp_dir}")
  os.makedirs(exp_dir, exist_ok=True)
  with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
    yaml.dump(cfg._asdict(), file)  # Add import for yaml at the top of the file



def log(cfg, metrics, micro_step, train_losses, valid_loss, optimizer, world_size):
  "Computes new metrcs and appends them to metrics. Logs on wandb. Prints log."
  # NOTE: train_losses is an array of losses, if DDP, this is from master_process only
  # NOTE: valid_loss is a float, already reduced across GPUs

  if isinstance(train_losses, list):
    train_loss = torch.stack(train_losses).mean().item() # avg loss

  new_metrics = {
    "micro_step": micro_step,
    "step": int(micro_step / cfg.grad_accumulation_steps),
    "tokens": micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
    "lr": optimizer.param_groups[0].get("lr", float("NaN")),
    "train/loss": train_loss,
    "train/ppl": math.exp(train_loss),
  }
  if valid_loss is not None:
    new_metrics["valid/loss"] = valid_loss
    new_metrics["valid/ppl"] = math.exp(valid_loss)

  for k,v in new_metrics.items():
    metrics[k].append(v)

  if cfg.print_progress:
    msg = ' | '.join(
      f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}"
      for key, value in new_metrics.items()
    )
    print(msg)
  
  if cfg.use_wandb:
    wandb.log(new_metrics)

"""   
def log_final_val_summary(cfg, val_loss):
  #Logs final validation loss summary
  if not cfg.use_wandb:
    return
  
  ppl = math.exp(val_loss)

  wandb.run.summary["final_validation_loss"] = val_loss
  wandb.run.summary["final_validation_ppl"] = ppl

  wandb.run.summary.update(
    {
      "final_validation_loss": val_loss,
      "final_validation_ppl": ppl,
    }
  )

  print_master(f"Final validation loss: {val_loss:.3e} | Final validation PPL: {ppl:.3e}")

"""   
def log_final_val_summary(cfg, val_loss):
    """Logs final validation loss summary"""
    if not cfg.use_wandb:
        print_master("Wandb not enabled, skipping final validation summary")
        return
    
    try:
        ppl = math.exp(val_loss)
        
        # This is the key change - add a metric specifically for bar charts
        # Use a flat key with no slashes, prefixed with "bar_"
        wandb.run.summary["bar_val_ppl"] = ppl
        wandb.run.summary["bar_val_loss"] = val_loss
        
        # Keep existing logging
        wandb.run.summary["final_validation_loss"] = val_loss
        wandb.run.summary["final_validation_ppl"] = ppl
        wandb.log({"final/validation_loss": val_loss, "final/validation_ppl": ppl})
        
        print_master(f"Logging final val metrics: loss={val_loss:.3e}, ppl={ppl:.3e}")
        print_master("Successfully logged final validation summary")
    except Exception as e:
        print_master(f"ERROR logging final validation summary: {str(e)}")

def log_final_train_summary(cfg, train_loss):
  """Logs final training loss summary"""
  if not cfg.use_wandb:
    return
  
  ppl = math.exp(train_loss)

  wandb.run.summary["final_training_loss"] = train_loss
  wandb.run.summary["final_training_ppl"] = ppl

  print_master(f"Final training loss: {train_loss:.3e} | Final training PPL: {ppl:.3e}")






