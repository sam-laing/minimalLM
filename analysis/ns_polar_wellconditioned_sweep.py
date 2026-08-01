"""
Whole-network resumed-training comparison: baseline (every param keeps the
checkpoint's normal ns_steps) vs. "polar-for-well-conditioned" (params whose
momentum condition number is BELOW --well_conditioned_threshold get exact
SVD-based polar orthogonalization; every other param keeps its normal
ns_steps unchanged). No freezing - every parameter trains normally, at the
config's real grad_accumulation_steps and lr, so this is much closer to
actually continuing training than the freeze-based isolation scripts
(analysis/ns_isolation_sweep.py, analysis/ns_group_isolation_sweep.py),
which forced grad_accumulation_steps=1 and only updated a small subset of
params at the full-network learning rate - a likely cause of the loss
increase seen there.

Mechanism: PyTorch optimizers' param_groups is just a list of dicts, and
Muon's momentum buffers live in optimizer.state, keyed by the parameter
object itself - not by which group it's in. So peeling a param out into a
new single-param group with orthogonalize=True (copying every other
hyperparam from its original group) is a safe, non-destructive
restructuring done right after loading the checkpoint's optimizer state; it
does not touch that param's already-loaded momentum buffer.

Usage:
  python analysis/ns_polar_wellconditioned_sweep.py \
    --config=config/config_runpod.yaml \
    --ckpt_path=/workspace/checkpoints/muon_90M_1p8B/ckpt_micro_step_6865.pth \
    --well_conditioned_threshold=1e4 \
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
from torch_utils import pytorch_setup

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'config/config_runpod.yaml', 'Base config (paths, model size, etc).')
flags.DEFINE_string('ckpt_path', None, 'Path to the checkpoint .pth to probe.')
flags.DEFINE_float('well_conditioned_threshold', 1e4,
                    'Switch params whose momentum condition number (max/min singular '
                    'value; worst sub-block for split QKV) is BELOW this to exact polar.')
flags.DEFINE_integer('num_iters', 100, 'Number of true optimizer steps to run per condition '
                                        '(each is grad_accumulation_steps micro-batches).')
flags.DEFINE_string('out_path', 'analysis/ns_polar_wellconditioned_results.json',
                     'Where to write the results JSON.')
flags.mark_flag_as_required('ckpt_path')


def build_engine(cfg, device, local_rank, ckpt):
  model, _ = construct_model(cfg)
  return TorchEngine(model, cfg, device, local_rank, ckpt)


def momentum_condition_number(buf):
  s = torch.linalg.svdvals(buf.float())
  return s[0].item() / max(s[-1].item(), 1e-12)


def rank_condition_numbers(model, optimizer):
  """Returns [(name, worst_cond, {sub: cond}), ...] sorted descending by worst_cond."""
  report = []
  for name, p in model.named_parameters():
    if p.ndim != 2:
      continue
    state = optimizer.state.get(p, {})
    if "q" in state and "k" in state and "v" in state:
      subs = {"q": state["q"], "k": state["k"], "v": state["v"]}
    else:
      subs = {"full": state}
    sub_conds = {}
    for sub_name, sub_state in subs.items():
      buf = sub_state.get("momentum_buffer") if isinstance(sub_state, dict) else None
      if buf is not None:
        sub_conds[sub_name] = momentum_condition_number(buf)
    if sub_conds:
      report.append((name, max(sub_conds.values()), sub_conds))
  report.sort(key=lambda x: -x[1])
  return report


def switch_well_conditioned_to_polar(engine, well_conditioned_names):
  """Peels each well-conditioned param out of its current group into a new
  single-param group with orthogonalize=True, copying every other hyperparam
  from the original group. Does not touch optimizer.state (momentum buffers),
  which is keyed by param object, not group membership."""
  named = dict(engine.model.named_parameters())
  targets = {named[n] for n in well_conditioned_names}

  new_groups = []
  for group in engine.optimizer.param_groups:
    remaining = []
    for p in group['params']:
      if p in targets:
        solo_group = {k: v for k, v in group.items() if k != 'params'}
        solo_group['params'] = [p]
        solo_group['orthogonalize'] = True
        new_groups.append(solo_group)
      else:
        remaining.append(p)
    group['params'] = remaining
  engine.optimizer.param_groups.extend(new_groups)


def run_condition(label, use_polar_for_well_conditioned, well_conditioned_names,
                   probe_cfg, ckpt, trainloader, device, local_rank, num_iters):
  print(f"\n=== {label} ===")
  engine = build_engine(probe_cfg, device, local_rank, ckpt)

  if use_polar_for_well_conditioned:
    switch_well_conditioned_to_polar(engine, well_conditioned_names)

  accumulation_steps = probe_cfg.grad_accumulation_steps
  rows = []
  data_iter = iter(trainloader)
  for step in range(num_iters):
    last_loss = None
    for _ in range(accumulation_steps):
      batch = next(data_iter)
      last_loss = engine.step(batch)
    rows.append({"step": step, "loss": float(last_loss)})
    if step % 10 == 0:
      print(f"  step {step}: loss={float(last_loss):.4f}")

  del engine
  torch.cuda.empty_cache()
  return rows


def main(_):
  base_cfg, _ = utils.load_config(FLAGS.config)
  local_rank, world_size, device, master_process = pytorch_setup(base_cfg)

  ckpt = torch.load(FLAGS.ckpt_path, map_location=device)

  # no freezing, no forced grad_accumulation_steps=1: every param trains
  # normally at the config's real accumulation cadence and lr
  probe_cfg = base_cfg._replace(
    resume=True,
    torch_compile=False,
    use_wandb=False,
    save_intermediate_checkpoints=False,
    save_last_checkpoint=False,
  )

  scan_engine = build_engine(probe_cfg, device, local_rank, ckpt)
  report = rank_condition_numbers(scan_engine.model, scan_engine.optimizer)

  well_conditioned = [n for n, w, _ in report if w < FLAGS.well_conditioned_threshold]
  ill_conditioned_count = len(report) - len(well_conditioned)

  print(f"well_conditioned_threshold={FLAGS.well_conditioned_threshold:g}")
  print(f"{len(well_conditioned)} / {len(report)} params below threshold -> switched to exact polar")
  print(f"{ill_conditioned_count} / {len(report)} params keep normal ns_steps unchanged\n")
  for name, worst, subs in report:
    flag = "  <-- POLAR" if worst < FLAGS.well_conditioned_threshold else ""
    print(f"  {worst:14.1f}  {name}  {subs}{flag}")

  if not well_conditioned:
    raise RuntimeError(
      f"No params were below well_conditioned_threshold={FLAGS.well_conditioned_threshold:g} "
      "- raise --well_conditioned_threshold and try again."
    )
  del scan_engine
  torch.cuda.empty_cache()

  trainloader, _ = get_dataloaders(probe_cfg)
  os.makedirs(os.path.dirname(FLAGS.out_path) or '.', exist_ok=True)

  results = {
    "ckpt_path": FLAGS.ckpt_path,
    "well_conditioned_threshold": FLAGS.well_conditioned_threshold,
    "well_conditioned_params": well_conditioned,
    "grad_accumulation_steps": probe_cfg.grad_accumulation_steps,
    "conditioning_report": [
      {"layer": n, "worst_condition_number": w, "sub_condition_numbers": s} for n, w, s in report
    ],
    "runs": {},
  }

  for label, use_polar in [("baseline", False), ("polar_for_well_conditioned", True)]:
    rows = run_condition(label, use_polar, well_conditioned, probe_cfg, ckpt,
                          trainloader, device, local_rank, FLAGS.num_iters)
    loss_reduction = rows[0]["loss"] - rows[-1]["loss"]
    results["runs"][label] = {
      "loss_start": rows[0]["loss"], "loss_end": rows[-1]["loss"],
      "loss_reduction": loss_reduction, "rows": rows,
    }
    print(f"  loss: {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}  (reduction={loss_reduction:+.4f})")

  with open(FLAGS.out_path, 'w') as f:
    json.dump(results, f, indent=2)
  print(f"\nsaved -> {FLAGS.out_path}")

  print("\n=== Summary ===")
  for label, run in results["runs"].items():
    print(f"  {label}: {run['loss_start']:.4f} -> {run['loss_end']:.4f}  reduction={run['loss_reduction']:+.4f}")


if __name__ == "__main__":
  app.run(main)
