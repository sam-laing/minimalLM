"""Pretrain a Transformer on language modeling."""

from absl import app, flags
from collections import defaultdict

import utils
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from data import get_dataloaders
from checkpoint_utils import save_checkpoint, maybe_load_checkpoint
from models import construct_model
from engine import TorchEngine
from optim.quant_adam.AOAdam import _AdamBase

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
  
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
  
  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  print(cfg)
  
  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
  
  # Load checkpoint and starting step
  ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)
  
  # Model
  model, model_cfg = construct_model(cfg)
  
  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  optimizer = engine.optimizer
  if cfg.log_quant_error:
    assert isinstance(_AdamBase, optimizer), "Only AOAdam is supported for logging quantization error" 
    optimizer_state = optimizer.state_dict()


  
  # Training
  print_master("=== Start Training! ===")
  metrics = defaultdict(list)
  train_losses = []
  
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start+1):
    step = micro_step // cfg.grad_accumulation_steps
    if step > cfg.steps_budget:
      break

    # Train
    train_loss = engine.step(micro_batch)
    train_losses.append(train_loss)

    # Eval
    valid_loss = None
    if cfg.eval and step % cfg.eval_every_steps == 0:
      print_master("Evaluating on validation set")
      valid_loss = engine.eval(validloader)
    
    # Log
    if step % cfg.log_every_steps == 0:
      if master_process:
        utils.log(cfg, metrics, micro_step, train_losses, valid_loss, engine.optimizer, world_size)
      train_losses = []
      #if cfg.log_quant_error:
      #  optimizer = engine.optimizer
        # still need to log ....

    
    # Checkpoint
    if master_process and cfg.save_intermediate_checkpoints \
        and micro_step % cfg.save_every_steps == 0:
      save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # End of training: log and save checkpoint
  print_master(f"=== Training Completed! ===")
  if master_process and cfg.save_last_checkpoint:
    save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
