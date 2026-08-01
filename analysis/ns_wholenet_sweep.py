"""
Whole-network NS-depth sweep on the ill-conditioned layers.

Selects every 2D Muon param whose momentum condition number exceeds
--condition_threshold, then for each condition in {ns2, ns3, ns5, ns10,
exact_polar} (--ns_steps_sweep), runs --num_iters *true optimizer steps* of
NORMAL whole-network training (every param updates, real
grad_accumulation_steps and lr from config - nothing frozen) except the
selected set's orthogonalization is overridden to that condition's value;
every other param keeps its normal ns_steps unchanged. Logs the loss
trajectory and net reduction per condition.

This is the ns2/3/5/10/exact_polar sweep from ns_group_isolation_sweep.py,
but using the no-freeze whole-network mechanism from
ns_polar_wellconditioned_sweep.py instead of freezing every other param -
freezing most of the network while keeping the full-network lr was making
loss increase for reasons unrelated to NS depth, confounding that result.

Mechanism: PyTorch optimizers' param_groups is just a list of dicts, and
Muon's momentum buffers live in optimizer.state, keyed by the parameter
object itself, not by which group it's in. Peeling the selected params out
into new single-param groups (copying every other hyperparam from their
original group, only overriding ns_steps/orthogonalize) is a safe,
non-destructive restructuring right after loading the checkpoint's
optimizer state; it does not touch their already-loaded momentum buffers.

Usage:
  python analysis/ns_wholenet_sweep.py \
    --config=config/config_runpod.yaml \
    --ckpt_path=/workspace/checkpoints/muon_90M_1p8B/ckpt_micro_step_6865.pth \
    --condition_threshold=1e5 \
    --ns_steps_sweep=2,3,5,10 \
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
flags.DEFINE_float('condition_threshold', 1e5,
                    'Select params whose momentum condition number (max/min singular '
                    'value; worst sub-block for split QKV) exceeds this.')
flags.DEFINE_string('ns_steps_sweep', '2,3,5,10', 'Comma-separated ns_steps values to compare.')
flags.DEFINE_boolean('include_exact_polar', True,
                      'Also run one condition using exact SVD-based polar (optim/muon.py '
                      'orthogonalize=True) on the selected set.')
flags.DEFINE_integer('num_iters', 100, 'Number of true optimizer steps to run per condition '
                                        '(each is grad_accumulation_steps micro-batches).')
flags.DEFINE_string('out_path', 'analysis/ns_wholenet_sweep_results.json',
                     'Where to write the results JSON.')
flags.mark_flag_as_required('ckpt_path')


def build_engine(cfg, device, local_rank, ckpt):
  model, _ = construct_model(cfg)
  return TorchEngine(model, cfg, device, local_rank, ckpt)


def momentum_condition_number(buf):
  s = torch.linalg.svdvals(buf.float())
  return s[0].item() / max(s[-1].item(), 1e-12)


def select_ill_conditioned_params(model, optimizer, threshold):
  """Returns (selected_names, report); report is [(name, worst_cond, {sub: cond}), ...]
  sorted descending by worst_cond."""
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
  selected = [name for name, worst, _ in report if worst > threshold]
  return selected, report


def override_selected_group(engine, selected_names, ns=None, use_exact_polar=False):
  """Peels each selected param out of its current group into a new
  single-param group with ns_steps=ns or orthogonalize=True, copying every
  other hyperparam from the original group. Does not touch optimizer.state
  (momentum buffers), which is keyed by param object, not group membership."""
  named = dict(engine.model.named_parameters())
  targets = {named[n] for n in selected_names}

  new_groups = []
  for group in engine.optimizer.param_groups:
    remaining = []
    for p in group['params']:
      if p in targets:
        solo_group = {k: v for k, v in group.items() if k != 'params'}
        solo_group['params'] = [p]
        if use_exact_polar:
          solo_group['orthogonalize'] = True
        else:
          solo_group['ns_steps'] = ns
        new_groups.append(solo_group)
      else:
        remaining.append(p)
    group['params'] = remaining
  engine.optimizer.param_groups.extend(new_groups)


def run_condition(label, ns, use_exact_polar, selected_names, probe_cfg, ckpt,
                   trainloader, device, local_rank, num_iters):
  print(f"\n=== {label} ===")
  engine = build_engine(probe_cfg, device, local_rank, ckpt)
  override_selected_group(engine, selected_names, ns=ns, use_exact_polar=use_exact_polar)

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
  # normally at the config's real accumulation cadence and lr; only the
  # selected set's orthogonalization method is overridden per condition
  probe_cfg = base_cfg._replace(
    resume=True,
    torch_compile=False,
    use_wandb=False,
    save_intermediate_checkpoints=False,
    save_last_checkpoint=False,
  )

  scan_engine = build_engine(probe_cfg, device, local_rank, ckpt)
  selected, report = select_ill_conditioned_params(
    scan_engine.model, scan_engine.optimizer, FLAGS.condition_threshold)

  print(f"condition_threshold={FLAGS.condition_threshold:g}")
  print(f"Selected {len(selected)} / {len(report)} params:")
  for name, worst, subs in report:
    flag = "  <-- SELECTED" if worst > FLAGS.condition_threshold else ""
    print(f"  {worst:14.1f}  {name}  {subs}{flag}")

  if not selected:
    raise RuntimeError(
      f"No params exceeded condition_threshold={FLAGS.condition_threshold:g} "
      "- lower --condition_threshold and try again."
    )
  del scan_engine
  torch.cuda.empty_cache()

  trainloader, _ = get_dataloaders(probe_cfg)
  os.makedirs(os.path.dirname(FLAGS.out_path) or '.', exist_ok=True)

  conditions = [(f"ns{ns}", ns, False) for ns in (int(x) for x in FLAGS.ns_steps_sweep.split(','))]
  if FLAGS.include_exact_polar:
    conditions.append(("exact_polar", None, True))

  results = {
    "ckpt_path": FLAGS.ckpt_path,
    "condition_threshold": FLAGS.condition_threshold,
    "selected_params": selected,
    "grad_accumulation_steps": probe_cfg.grad_accumulation_steps,
    "conditioning_report": [
      {"layer": n, "worst_condition_number": w, "sub_condition_numbers": s} for n, w, s in report
    ],
    "runs": {},
  }

  summary = []
  for label, ns, use_exact_polar in conditions:
    rows = run_condition(label, ns, use_exact_polar, selected, probe_cfg, ckpt,
                          trainloader, device, local_rank, FLAGS.num_iters)
    loss_reduction = rows[0]["loss"] - rows[-1]["loss"]
    results["runs"][label] = {
      "ns_steps": ns, "orthogonalize_exact": use_exact_polar,
      "loss_start": rows[0]["loss"], "loss_end": rows[-1]["loss"],
      "loss_reduction": loss_reduction, "rows": rows,
    }
    summary.append((label, rows[0]["loss"], rows[-1]["loss"], loss_reduction))
    print(f"  loss: {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}  (reduction={loss_reduction:+.4f})")

  with open(FLAGS.out_path, 'w') as f:
    json.dump(results, f, indent=2)
  print(f"\nsaved -> {FLAGS.out_path}")

  print("\n=== Summary (loss_start -> loss_end, reduction) ===")
  for label, l0, l1, red in summary:
    print(f"  {label}: {l0:.4f} -> {l1:.4f}  reduction={red:+.4f}")


if __name__ == "__main__":
  app.run(main)
