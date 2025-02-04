"""Pretrain a Transformer on language modeling."""

from absl import app, flags
from collections import defaultdict

import utils
import torch_utils 
from torch_utils import get_moments_dict, get_histogram_for_adam_layer, save_layer_histogram_plots, save_all_layer_histogram_plots, compute_kl_div_and_ks_test_for_all_layers
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from data import get_dataloaders
from checkpoint_utils import save_checkpoint, maybe_load_checkpoint
from models import construct_model
from engine import TorchEngine
import json


flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
  
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
  
  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  print(vars(cfg))
  
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

  utils.maybe_make_folders_for_plots(model, cfg.path_for_moment_plots)
  
  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  # Training
  print_master("=== Start Training! ===")
  metrics = defaultdict(list)
  train_losses = []


  moments_json = {}
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start+1):
    step = micro_step // cfg.grad_accumulation_steps
    if step > cfg.steps_budget:
      break

    # Train
    train_loss = engine.step(micro_batch)
    train_losses.append(train_loss)

    # get moments, kl divergence, and hypothesis test of each layer

    optimizer = engine.optimizer
    moment_dict = get_moments_dict(model, optimizer)
    if micro_step % cfg.do_plots_every_steps == 0:
      save_layer_histogram_plots(model, moment_dict, cfg.path_for_moment_plots, micro_step)
    
    test_stats = compute_kl_div_and_ks_test_for_all_layers(moment_dict)
    print("\n", test_stats, "\n")

    moments_json[micro_step] = test_stats





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
    
    # Checkpoint
    if master_process and cfg.save_intermediate_checkpoints \
        and micro_step % cfg.save_every_steps == 0:
      save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # End of training: log and save checkpoint
  print_master(f"=== Training Completed! ===")

  # save the moments json
  json.dump(moments_json, open(cfg.path_for_moment_plots + "/moments.json", "w"))

  if master_process and cfg.save_last_checkpoint:
    save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
