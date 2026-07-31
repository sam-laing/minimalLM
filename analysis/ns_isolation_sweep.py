"""
Isolated Newton-Schulz depth sweep.

Given a checkpoint, finds the matrix parameter(s) with the most ill-conditioned
Muon momentum buffer (lowest normalized stable rank), freezes every other
parameter, and continues training *only* that parameter from the checkpoint's
exact momentum state for a short burst, once per ns_steps value in the sweep
(plus an exact-SVD-polar condition, the ns_steps->inf reference point).

Because Muon's momentum buffer is updated *before* Newton-Schulz is applied
(see optim/muon.py:_muon_step), freezing every other param cleanly isolates
"how much does NS depth change this one parameter's trajectory" without any
of the other params' dynamics leaking in.

All sweep runs replay the exact same batches (fresh sequential DataLoader
iterator per run) so differences are attributable to ns_steps, not data order.

Usage:
  python analysis/ns_isolation_sweep.py \
    --config=config/config_runpod.yaml \
    --ckpt_path=/workspace/checkpoints/muon_90M_1p8B/ckpt_micro_step_13731.pth \
    --num_targets=3 \
    --ns_steps_sweep=1,2,5,10,20 \
    --num_iters=100
"""

import json
import os

import torch
from absl import app, flags

import utils
from data import get_dataloaders
from engine import TorchEngine
from models import construct_model
from moment_utils import rank_momentum_conditioning, stable_rank
from torch_utils import pytorch_setup

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'config/config_runpod.yaml', 'Base config (paths, model size, etc).')
flags.DEFINE_string('ckpt_path', None, 'Path to the checkpoint .pth to probe.')
flags.DEFINE_string('target_param', 'auto',
                     "Comma-separated param names to isolate, or 'auto' to pick the "
                     "--num_targets most ill-conditioned automatically.")
flags.DEFINE_integer('num_targets', 1, "How many worst-conditioned params to probe when target_param='auto'.")
flags.DEFINE_string('ns_steps_sweep', '1,2,5,10,20', 'Comma-separated ns_steps values to compare.')
flags.DEFINE_boolean('include_exact_polar', True,
                      'Also run one condition using the exact SVD-based polar factor '
                      '(optim/muon.py orthogonalize=True) as the ns_steps->inf reference point.')
flags.DEFINE_integer('num_iters', 100, 'Number of optimizer steps to run per ns_steps value.')
flags.DEFINE_string('out_path', 'analysis/ns_isolation_results.json', 'Where to write the results JSON.')
flags.mark_flag_as_required('ckpt_path')


def build_engine(cfg, device, local_rank, ckpt):
  model, _ = construct_model(cfg)
  return TorchEngine(model, cfg, device, local_rank, ckpt)


def run_one_target(target_name, probe_cfg, ckpt, trainloader, device, local_rank, conditions):
  """Runs every (ns_steps / exact_polar) condition for a single isolated target param."""
  runs = {}
  summary = []
  for label, ns, use_exact_polar in conditions:
    print(f"  --- {label} ---")
    engine = build_engine(probe_cfg, device, local_rank, ckpt)

    named = dict(engine.model.named_parameters())
    target_param = named[target_name]
    for name, p in named.items():
      p.requires_grad_(name == target_name)

    # ns_steps/orthogonalize are read per-param-group (optim/muon.py:218-235);
    # override only the group holding the target param. Every other param in
    # that group is frozen above, so it never gets a grad and never gets
    # stepped - this override only ever touches the isolated parameter.
    for group in engine.optimizer.param_groups:
      if any(p is target_param for p in group['params']):
        if use_exact_polar:
          group['orthogonalize'] = True
        else:
          group['ns_steps'] = ns

    rows = []
    data_iter = iter(trainloader)
    for it in range(FLAGS.num_iters):
      batch = next(data_iter)
      loss = engine.step(batch)

      sr = None
      state = engine.optimizer.state.get(target_param, {})
      if 'momentum_buffer' in state:
        sr = stable_rank(state['momentum_buffer'])

      rows.append({'iter': it, 'loss': float(loss), 'stable_rank': sr})
      if it % 25 == 0:
        print(f"    iter {it}: loss={float(loss):.4f} stable_rank={sr}")

    runs[label] = {'ns_steps': ns, 'orthogonalize_exact': use_exact_polar, 'rows': rows}
    summary.append((label, rows[-1]['loss'], rows[-1]['stable_rank']))

    del engine
    torch.cuda.empty_cache()

  return runs, summary


def main(_):
  base_cfg, _ = utils.load_config(FLAGS.config)
  local_rank, world_size, device, master_process = pytorch_setup(base_cfg)

  ckpt = torch.load(FLAGS.ckpt_path, map_location=device)

  # short, undistracted probe: one real optimizer step per iter, no compile
  # overhead per sweep value, no checkpointing/wandb noise
  probe_cfg = base_cfg._replace(
    resume=True,
    grad_accumulation_steps=1,
    torch_compile=False,
    use_wandb=False,
    save_intermediate_checkpoints=False,
    save_last_checkpoint=False,
  )

  # scan the checkpoint once to find (or validate) the isolation target(s)
  scan_engine = build_engine(probe_cfg, device, local_rank, ckpt)
  ranked = rank_momentum_conditioning(scan_engine.model, scan_engine.optimizer)
  print("Momentum buffer conditioning (lower normalized stable rank = more ill-conditioned):")
  for name, score in ranked[:10]:
    print(f"  {score:.4f}  {name}")
  print("  ...")
  for name, score in ranked[-3:]:
    print(f"  {score:.4f}  {name}  (well-conditioned)")

  if FLAGS.target_param == 'auto':
    target_names = [name for name, _ in ranked[:FLAGS.num_targets]]
  else:
    target_names = [t.strip() for t in FLAGS.target_param.split(',')]
  print(f"\nIsolating {len(target_names)} target(s): {target_names}\n")
  del scan_engine
  torch.cuda.empty_cache()

  trainloader, _ = get_dataloaders(probe_cfg)

  os.makedirs(os.path.dirname(FLAGS.out_path) or '.', exist_ok=True)

  conditions = [(f"ns{ns}", ns, False) for ns in (int(x) for x in FLAGS.ns_steps_sweep.split(','))]
  if FLAGS.include_exact_polar:
    conditions.append(("exact_polar", None, True))

  results = {
    'ckpt_path': FLAGS.ckpt_path,
    'target_params': target_names,
    'conditioning_ranking': ranked,  # [(name, normalized_stable_rank), ...], most ill-conditioned first
    'targets': {},
  }

  all_summaries = {}
  for target_name in target_names:
    print(f"=== target: {target_name} ===")
    runs, summary = run_one_target(target_name, probe_cfg, ckpt, trainloader, device, local_rank, conditions)
    results['targets'][target_name] = runs
    all_summaries[target_name] = summary

  with open(FLAGS.out_path, 'w') as f:
    json.dump(results, f, indent=2)
  print(f"\nsaved -> {FLAGS.out_path}")

  print("\n=== Summary (final loss, final stable_rank) ===")
  for target_name, summary in all_summaries.items():
    print(f"target: {target_name}")
    for label, final_loss, final_sr in summary:
      print(f"  {label}: loss={final_loss:.4f} stable_rank={final_sr:.4f}")


if __name__ == "__main__":
  app.run(main)
