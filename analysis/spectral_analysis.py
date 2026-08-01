"""
Full spectral-conditioning study of Muon's gradient/momentum matrices across
training, per the spec: practical conditioning (ordinary + quantile condition
numbers), tiny-mode fractions, spectral flatness, gradient-vs-momentum
comparison, exact-polar-vs-finite-NS convergence (delta_q, per-mode
multipliers, energy fractions), and layer-specific "how many NS steps are
enough" guidance.

Auto-discovers whatever checkpoints actually exist in --ckpt_dir (no
hardcoded/arbitrary step numbers) and labels each by its fraction of the
configured training budget.

Key simplification (verified algebraically, not assumed): Newton-Schulz's
update X_{k+1} = a*X + (b*A + c*A@A)@X with A = X@X.T preserves X's singular
vectors exactly and evolves each singular value independently via the scalar
quintic map phi(x) = a*x + b*x^3 + c*x^5. So instead of doing q matrix
multiplications per mode, we SVD once and iterate phi on the singular value
sequence - exact, not an approximation, and far cheaper.

Usage:
  python analysis/spectral_analysis.py \
    --config=config/config_runpod.yaml \
    --ckpt_dir=/workspace/checkpoints/muon_90M_1p8B \
    --out_path=analysis/spectral_results.csv
"""

import glob
import os
import re

import torch
from absl import app, flags

import utils
from models import construct_model, get_param_groups
from optim import intialize_optimizer

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'config/config_runpod.yaml', 'Base config (paths, model size, etc).')
flags.DEFINE_string('ckpt_dir', None, 'Directory containing ckpt_micro_step_*.pth files.')
flags.DEFINE_string('out_path', 'analysis/spectral_results.csv', 'Where to write the flat results CSV.')
flags.DEFINE_string('json_out_path', 'analysis/spectral_results.json',
                     'Where to write the nested JSON (per checkpoint -> per layer -> '
                     'momentum/gradient side by side, for direct comparison).')
flags.mark_flag_as_required('ckpt_dir')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# exact constants from optim/muon.py's zeropower_via_newtonschulz5
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_EPS = 1e-7

NS_Q_VALUES = [2, 3, 5, 10, 15]
TAU_VALUES = [1e-2, 1e-3, 1e-4]
FRAC_VALUES = [0.5, 0.9]
CONVERGENCE_TOL = 0.1
MAX_Q_SCAN = 30

FIELDNAMES = (
  ["checkpoint_step", "pct_of_training", "layer", "matrix_type", "r",
   "condition_number", "kappa_50", "kappa_90", "flatness_F"]
  + [f"T_{tau:g}" for tau in TAU_VALUES]
  + [f"delta_{q}" for q in NS_Q_VALUES]
  # post-NS singular value distribution at each depth q (exact polar's
  # distribution is degenerate: every mode maps to exactly 1.0) - shows how
  # far NS_q's actual output spectrum still is from that flat target, and
  # specifically how the worst (tail) mode lags the median
  + [f"sv_min_ns{q}" for q in NS_Q_VALUES]
  + [f"sv_median_ns{q}" for q in NS_Q_VALUES]
  + [f"sv_max_ns{q}" for q in NS_Q_VALUES]
  + [f"energy_tiny_frac_polar_tau{tau:g}" for tau in TAU_VALUES]
  + [f"energy_tiny_frac_ns{q}_tau{tau:g}" for q in NS_Q_VALUES for tau in TAU_VALUES]
  + [f"min_q_top{int(f*100)}pct" for f in FRAC_VALUES]
)


def discover_checkpoints(ckpt_dir):
  paths = glob.glob(os.path.join(ckpt_dir, "ckpt_micro_step_*.pth"))
  steps = sorted(int(re.search(r"ckpt_micro_step_(\d+)\.pth", p).group(1)) for p in paths)
  return steps


def phi(x):
  """Scalar Newton-Schulz map: phi(x) = a*x + b*x^3 + c*x^5."""
  return NS_A * x + NS_B * x**3 + NS_C * x**5


def phi_q(x, q):
  """Iterate phi q times."""
  for _ in range(q):
    x = phi(x)
  return x


def spectral_row(name, matrix_type, buf, step, pct):
  buf = buf.to(DEVICE).float()
  s = torch.linalg.svdvals(buf)  # descending
  r = s.numel()
  sigma1 = s[0].item()

  # --- metric 1: practical spectral conditioning ---
  idx50 = min(r - 1, int(torch.ceil(torch.tensor(0.5 * r)).item()) - 1)
  idx90 = min(r - 1, int(torch.ceil(torch.tensor(0.9 * r)).item()) - 1)
  condition_number = (sigma1 / max(s[-1].item(), 1e-12))
  kappa_50 = (sigma1 / max(s[idx50].item(), 1e-12))
  kappa_90 = (sigma1 / max(s[idx90].item(), 1e-12))

  # --- metric 2: tiny-mode fractions ---
  ratios = s / sigma1
  tiny_fracs = {tau: (ratios < tau).float().mean().item() for tau in TAU_VALUES}

  # --- metric 3: spectral flatness / normalized nuclear rank ---
  nuclear = s.sum()
  frob_sq = (s**2).sum()
  flatness_F = (nuclear**2 / (r * frob_sq)).item()

  # --- normalized singular values, matching training's normalization
  # exactly: X /= (X.norm() + eps), i.e. divide by the FULL Frobenius norm ---
  frob_norm = buf.norm().item()
  sigma_tilde = (s / (frob_norm + NS_EPS))

  row = {
    "checkpoint_step": step, "pct_of_training": pct, "layer": name,
    "matrix_type": matrix_type, "r": r,
    "condition_number": condition_number, "kappa_50": kappa_50, "kappa_90": kappa_90,
    "flatness_F": flatness_F,
  }
  for tau in TAU_VALUES:
    row[f"T_{tau:g}"] = tiny_fracs[tau]

  # --- metric 5: exact polar vs finite NS ---
  # D_inf (exact polar) gives every mode multiplier 1 -> ||D_inf||_F^2 = r.
  # delta_q = ||D_q - D_inf||_F / ||D_inf||_F = sqrt(mean((phi_q(sigma_tilde_i) - 1)^2))
  # (both D_q and D_inf share U,V, so this Frobenius-norm ratio reduces to a
  # pure function of the per-mode multiplier sequence - see module docstring)
  phi_q_vals = {}
  for q in NS_Q_VALUES:
    pv = phi_q(sigma_tilde, q)
    phi_q_vals[q] = pv
    delta_q = torch.sqrt(((pv - 1.0) ** 2).mean()).item()
    row[f"delta_{q}"] = delta_q
    # post-NS singular value distribution (target: all == 1.0 for exact polar)
    row[f"sv_min_ns{q}"] = pv.min().item()
    row[f"sv_median_ns{q}"] = pv.median().item()
    row[f"sv_max_ns{q}"] = pv.max().item()

  # --- energy fraction assigned to tiny modes: exact polar vs NS_q ---
  for tau in TAU_VALUES:
    tiny_mask = ratios < tau
    # exact polar: every mode contributes equally (multiplier 1) -> energy
    # fraction to tiny modes is just their count fraction
    row[f"energy_tiny_frac_polar_tau{tau:g}"] = tiny_mask.float().mean().item()
    for q in NS_Q_VALUES:
      pv = phi_q_vals[q]
      total_energy = (pv**2).sum().item()
      tiny_energy = (pv[tiny_mask] ** 2).sum().item()
      row[f"energy_tiny_frac_ns{q}_tau{tau:g}"] = (tiny_energy / total_energy) if total_energy > 0 else float('nan')

  # --- metric 6: minimum q so that top f% of modes (by magnitude) are within
  # CONVERGENCE_TOL of the exact-polar multiplier (1.0) ---
  # per-mode: smallest q in [1, MAX_Q_SCAN] with |phi_q(sigma_tilde_i) - 1| <= tol
  converged_at = torch.full((r,), float('inf'))
  x = sigma_tilde.clone()
  still_open = torch.ones(r, dtype=torch.bool)
  for q in range(1, MAX_Q_SCAN + 1):
    x = phi(x)
    newly_converged = still_open & (torch.abs(x - 1.0) <= CONVERGENCE_TOL)
    converged_at[newly_converged] = q
    still_open &= ~newly_converged
    if not still_open.any():
      break

  for f in FRAC_VALUES:
    top_n = max(1, int((f * r) + 0.9999999))  # ceil
    # "top" = largest singular values = first top_n entries (s is descending)
    needed = converged_at[:top_n]
    min_q = needed.max().item()  # worst-case among that fraction
    row[f"min_q_top{int(f*100)}pct"] = min_q

  return row


def main(_):
  base_cfg, _ = utils.load_config(FLAGS.config)
  steps = discover_checkpoints(FLAGS.ckpt_dir)
  if not steps:
    raise RuntimeError(f"No ckpt_micro_step_*.pth files found in {FLAGS.ckpt_dir}")
  total_micro_steps = base_cfg.steps_budget * base_cfg.grad_accumulation_steps
  print(f"device={DEVICE}")
  print(f"found {len(steps)} checkpoints in {FLAGS.ckpt_dir}: {steps}")
  print(f"total_micro_steps (from config)={total_micro_steps}")

  model, _ = construct_model(base_cfg)
  param_groups = get_param_groups(model, base_cfg.weight_decay)
  optimizer = intialize_optimizer(param_groups, base_cfg)
  matrix_params = {n: p for n, p in model.named_parameters() if p.ndim == 2}
  print(f"tracking {len(matrix_params)} matrix params (qkv split further at read time)")

  os.makedirs(os.path.dirname(FLAGS.out_path) or '.', exist_ok=True)
  os.makedirs(os.path.dirname(FLAGS.json_out_path) or '.', exist_ok=True)

  import csv
  import json

  # nested structure: checkpoint -> layer -> {momentum: {...}, gradient: {...}}
  # for direct side-by-side comparison, per the "gradient vs momentum" spec item
  json_results = {"checkpoints": []}

  with open(FLAGS.out_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    writer.writeheader()

    for step in steps:
      pct = round(100 * step / total_micro_steps, 1)
      print(f"\n=== checkpoint {step} ({pct}% of training) ===")
      path = os.path.join(FLAGS.ckpt_dir, f"ckpt_micro_step_{step}.pth")
      ckpt = torch.load(path, map_location="cpu", weights_only=False)
      optimizer.load_state_dict(ckpt["optimizer"])

      raw_grads = ckpt.get("grads")
      grads = None
      if raw_grads is not None:
        grads = {k.removeprefix("_orig_mod."): v for k, v in raw_grads.items()}
      else:
        print(f"  WARNING: checkpoint {step} has no saved grads, skipping gradient metrics")

      ckpt_entry = {"checkpoint_step": step, "pct_of_training": pct, "layers": {}}

      for name, p in matrix_params.items():
        # --- momentum buffer(s), split q/k/v the way Muon's sep_qkv stores them ---
        state = optimizer.state.get(p, {})
        if "q" in state and "k" in state and "v" in state:
          mom_subs = {"q": state["q"], "k": state["k"], "v": state["v"]}
        else:
          mom_subs = {"full": state}
        for sub_name, sub_state in mom_subs.items():
          buf = sub_state.get("momentum_buffer") if isinstance(sub_state, dict) else None
          if buf is None:
            continue
          full_name = f"{name}.{sub_name}" if sub_name != "full" else name
          row = spectral_row(full_name, "momentum", buf, step, pct)
          writer.writerow(row)
          ckpt_entry["layers"].setdefault(full_name, {})["momentum"] = {
            k: v for k, v in row.items() if k not in ("checkpoint_step", "pct_of_training", "layer", "matrix_type")
          }

        # --- raw gradient, manually split q/k/v to match, since it's saved fused ---
        if grads is not None:
          g = grads.get(name)
          if g is not None:
            if g.ndim == 2 and g.size(0) == 3 * g.size(1):
              d = g.size(1)
              g_q, g_k, g_v = g.split(d, dim=0)
              grad_subs = {"q": g_q, "k": g_k, "v": g_v}
            else:
              grad_subs = {"full": g}
            for sub_name, gbuf in grad_subs.items():
              full_name = f"{name}.{sub_name}" if sub_name != "full" else name
              row = spectral_row(full_name, "gradient", gbuf, step, pct)
              writer.writerow(row)
              ckpt_entry["layers"].setdefault(full_name, {})["gradient"] = {
                k: v for k, v in row.items() if k not in ("checkpoint_step", "pct_of_training", "layer", "matrix_type")
              }

      csvfile.flush()
      json_results["checkpoints"].append(ckpt_entry)
      print(f"  checkpoint {step}: done")

  with open(FLAGS.json_out_path, 'w') as jf:
    json.dump(json_results, jf, indent=2)

  print(f"\nsaved -> {FLAGS.out_path}")
  print(f"saved -> {FLAGS.json_out_path}")


if __name__ == "__main__":
  app.run(main)
