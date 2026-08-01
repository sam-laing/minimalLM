"""
Group NS-depth ablation.

Selects every 2D Muon param whose momentum buffer's raw condition number
(sigma_max/sigma_min, worst sub-block for split QKV) exceeds --threshold,
freezes every other parameter, and resumes training *only* the selected set
from the checkpoint's exact momentum state for --num_iters steps, once per
condition (ns_steps in --ns_steps_sweep, plus exact SVD-based polar as the
ns_steps->inf reference). Logs the full loss trajectory and net loss
reduction (loss[0] - loss[-1]) for each condition.

Because Muon's momentum buffer is updated *before* Newton-Schulz is applied
(optim/muon.py:_muon_step), and frozen params receive no gradient and are
skipped entirely by the optimizer step (`if p.grad is None: continue`),
overriding ns_steps/orthogonalize globally on every param group is safe: it
only ever actually affects the unfrozen (selected) params in practice.

All conditions replay the exact same batch sequence (fresh sequential
DataLoader iterator per run) so differences are attributable to
orthogonalization depth, not data order.

Usage:
  python analysis/ns_group_isolation_sweep.py \
    --config=config/config_runpod.yaml \
    --ckpt_path=/workspace/checkpoints/muon_90M_1p8B/ckpt_micro_step_27471.pth \
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
                      'Also run one condition using the exact SVD-based polar factor '
                      '(optim/muon.py orthogonalize=True) as the ns_steps->inf reference.')
flags.DEFINE_integer('num_iters', 100, 'Number of optimizer steps to run per condition.')
flags.DEFINE_string('out_path', 'analysis/ns_group_isolation_results.json',
                     'Where to write the results JSON.')
flags.mark_flag_as_required('ckpt_path')


def build_engine(cfg, device, local_rank, ckpt):
  model, _ = construct_model(cfg)
  return TorchEngine(model, cfg, device, local_rank, ckpt)


def momentum_condition_number(buf):
  s = torch.linalg.svdvals(buf.float())
  return s[0].item() / max(s[-1].item(), 1e-12)


def select_ill_conditioned_params(model, optimizer, threshold):
  """Returns (selected_names, report) where report is [(name, worst_cond, {sub: cond}), ...]
  sorted by worst_cond descending. worst_cond is the max condition number across
  q/k/v sub-blocks (or the single momentum buffer, for non-QKV params)."""
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
    if not sub_conds:
      continue
    worst = max(sub_conds.values())
    report.append((name, worst, sub_conds))

  report.sort(key=lambda x: -x[1])
  selected = [name for name, worst, _ in report if worst > threshold]
  return selected, report


def main(_):
  base_cfg, _ = utils.load_config(FLAGS.config)
  local_rank, world_size, device, master_process = pytorch_setup(base_cfg)

  ckpt = torch.load(FLAGS.ckpt_path, map_location=device)

  probe_cfg = base_cfg._replace(
    resume=True,
    grad_accumulation_steps=1,
    torch_compile=False,
    use_wandb=False,
    save_intermediate_checkpoints=False,
    save_last_checkpoint=False,
  )

  # scan the checkpoint once to find the ill-conditioned set
  scan_engine = build_engine(probe_cfg, device, local_rank, ckpt)
  selected, report = select_ill_conditioned_params(
    scan_engine.model, scan_engine.optimizer, FLAGS.condition_threshold)

  print(f"Momentum condition numbers (threshold={FLAGS.condition_threshold:g}):")
  for name, worst, subs in report[:15]:
    flag = "  <-- SELECTED" if worst > FLAGS.condition_threshold else ""
    print(f"  {worst:14.1f}  {name}  {subs}{flag}")
  print(f"\nSelected {len(selected)} / {len(report)} params:")
  for n in selected:
    print(f"  {n}")
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
    "selected_layers": selected,
    "conditioning_report": [
      {"layer": n, "worst_condition_number": w, "sub_condition_numbers": s} for n, w, s in report
    ],
    "runs": {},
  }

  selected_set = set(selected)
  summary = []
  for label, ns, use_exact_polar in conditions:
    print(f"\n=== {label} ===")
    engine = build_engine(probe_cfg, device, local_rank, ckpt)

    named = dict(engine.model.named_parameters())
    for name, p in named.items():
      p.requires_grad_(name in selected_set)

    # frozen params get no gradient and are skipped by Muon's step loop
    # regardless of these settings, so overriding every group is safe and
    # only actually affects the selected (unfrozen) params in practice
    for group in engine.optimizer.param_groups:
      if use_exact_polar:
        group['orthogonalize'] = True
      else:
        group['ns_steps'] = ns

    rows = []
    data_iter = iter(trainloader)
    for it in range(FLAGS.num_iters):
      batch = next(data_iter)
      loss = engine.step(batch)
      rows.append({"iter": it, "loss": float(loss)})
      if it % 10 == 0:
        print(f"  iter {it}: loss={float(loss):.4f}")

    loss_reduction = rows[0]["loss"] - rows[-1]["loss"]
    results["runs"][label] = {
      "ns_steps": ns,
      "orthogonalize_exact": use_exact_polar,
      "loss_start": rows[0]["loss"],
      "loss_end": rows[-1]["loss"],
      "loss_reduction": loss_reduction,
      "rows": rows,
    }
    summary.append((label, rows[0]["loss"], rows[-1]["loss"], loss_reduction))
    print(f"  loss: {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}  (reduction={loss_reduction:+.4f})")

    del engine
    torch.cuda.empty_cache()

  with open(FLAGS.out_path, 'w') as f:
    json.dump(results, f, indent=2)
  print(f"\nsaved -> {FLAGS.out_path}")

  print("\n=== Summary (loss_start -> loss_end, reduction) ===")
  for label, l0, l1, red in summary:
    print(f"  {label}: {l0:.4f} -> {l1:.4f}  reduction={red:+.4f}")


if __name__ == "__main__":
  app.run(main)
