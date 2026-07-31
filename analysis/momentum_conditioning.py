"""Momentum-buffer conditioning study for the muon_90M_1p8B (no-rotation)
baseline run, at 9 representative checkpoints (first 3 / middle 3 / last 3
of the 59 saved).

For every matrix Muon optimizes (QKV kept split into q/k/v, matching how
Muon's own sep_qkv=True actually stores them - three separate 576x576
momentum buffers, not one 1728x576 buffer), computes singular-value-based
conditioning of:
  - "raw":       the momentum buffer itself, before any orthogonalization
  - "ns_1".."ns_5": the Newton-Schulz iterate after k steps, replicating
                  optim/muon.py's zeropower_via_newtonschulz5 exactly
                  (a=3.4445, b=-4.7750, c=2.0315, eps=1e-7, bfloat16),
                  so this is the conditioning trajectory through the same
                  iteration actually used in training (ns_steps=5)
  - "true_polar": the exact polar factor (full SVD, U @ Vh) as a reference
                  target, plus how far the 5-step NS result is from it
                  (relative Frobenius distance, dist_to_polar column)

For each stage: sigma_max, sigma_min, sigma_mean, sigma_std (spread),
condition_number, effective_rank.

Runs on GPU if available (set via CUDA_VISIBLE_DEVICES / salloc), falls
back to CPU otherwise.

Output layout - one directory per checkpoint, written/flushed as it goes:
  analysis/results/ckpt_<step>/conditioning.csv
  analysis/results/ckpt_<step>/singular_value_histograms/<layer>.pdf
    (raw / ns_1..ns_5 / true_polar histograms side by side, for a selected
    subset of layers: embedding, LM head, and attention q/k/v/out from the
    first and last transformer layer)
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import utils
from models import construct_model, get_param_groups
from optim import intialize_optimizer

CKPT_DIR = "/data/horse/ws/sala597i-slaing/checkpoints/muon_90M_1p8B"
CONFIG_PATH = "config/config.yaml"
OUT_ROOT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# first 3, middle 3, last 3 of the 59 available checkpoints for this run
CHECKPOINT_GROUPS = {
    "early": [471, 943, 1415],
    "mid": [13687, 14159, 14631],
    "late": [26903, 27375, 27452],
}

NS_STEPS_TO_RECORD = [1, 2, 3, 4, 5]
STAGES = ["raw", "normalized"] + [f"ns_{k}" for k in NS_STEPS_TO_RECORD] + ["true_polar"]
# what actually gets plotted: normalized starting point + the 5 NS iterates
# (raw and true_polar are still computed/recorded in the CSV, just not plotted)
PLOT_STAGES = ["normalized"] + [f"ns_{k}" for k in NS_STEPS_TO_RECORD]

# exact constants from optim/muon.py's zeropower_via_newtonschulz5 - verified
# against the current source, not assumed
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_EPS = 1e-7  # class default in optim/muon.py; init_optim.py never overrides it

# all 8 transformer layers' attention q/k/v/out, plus embedding/LM head
N_LAYERS = 8
PLOT_LAYERS = {"embed_tokens.weight", "lm_head.weight"}
for _l in range(N_LAYERS):
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_qkv.weight.q")
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_qkv.weight.k")
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_qkv.weight.v")
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_out.weight")

FIELDNAMES = [
    "layer", "stage",
    "sigma_max", "sigma_min", "sigma_mean", "sigma_std",
    "condition_number", "effective_rank", "dist_to_polar",
]


def spectrum_stats(s):
    smax, smin = s[0].item(), s[-1].item()
    return {
        "sigma_max": smax,
        "sigma_min": smin,
        "sigma_mean": s.mean().item(),
        "sigma_std": s.std().item(),
        "condition_number": smax / max(smin, 1e-12),
        "effective_rank": (s.sum() ** 2 / (s ** 2).sum()).item(),
    }


def newton_schulz_iterates(G, steps):
    """Replicates optim/muon.py's zeropower_via_newtonschulz5 exactly, but
    returns the normalized starting point (before any iteration) plus the
    iterate after every step, instead of only the final one.

    Note: normalization (X /= X.norm()) happens exactly once, here, before
    the loop - the loop itself never renormalizes at any step, confirmed
    against the actual optim/muon.py source.
    """
    X = G.to(DEVICE).bfloat16()
    X = X / (X.norm() + NS_EPS)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    normalized = X.T if transposed else X
    iterates = []
    for _ in range(steps):
        A = X @ X.T
        B = NS_B * A + NS_C * A @ A
        X = NS_A * X + B @ X
        iterates.append(X.T if transposed else X)
    return normalized, iterates


def true_polar(G):
    """Exact polar factor via full SVD - matches optim/muon.py's orthogonalise()."""
    Gc = G.to(DEVICE).float()
    transposed = False
    if Gc.size(0) > Gc.size(1):
        Gc = Gc.T
        transposed = True
    U, _, Vh = torch.linalg.svd(Gc, full_matrices=False)
    out = U @ Vh
    return out.T if transposed else out


def plot_histograms(name, singular_values_by_stage, out_path):
    fig, axes = plt.subplots(1, len(PLOT_STAGES), figsize=(3 * len(PLOT_STAGES), 3.5), sharey=False)
    for ax, stage in zip(axes, PLOT_STAGES):
        s = singular_values_by_stage[stage]
        smin, smax = s.min().item(), s.max().item()
        condition_number = smax / max(smin, 1e-12)  # standard definition: max/min
        ax.hist(s.cpu().numpy(), bins=40, color="tab:blue", alpha=0.8)
        ax.set_title(
            f"{stage}\ncond(max/min)={condition_number:.3g}\nmin={smin:.3g}  max={smax:.3g}",
            fontsize=8,
        )
        ax.set_xlabel("singular value")
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    axes[0].set_ylabel("count")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def analyze_matrix(name, buf, writer):
    buf = buf.to(DEVICE)
    singular_values_by_stage = {}

    s_raw = torch.linalg.svdvals(buf.float())
    singular_values_by_stage["raw"] = s_raw
    writer.writerow({"layer": name, "stage": "raw", "dist_to_polar": "",
                      **spectrum_stats(s_raw)})

    polar = true_polar(buf)
    s_polar = torch.linalg.svdvals(polar)
    singular_values_by_stage["true_polar"] = s_polar

    normalized, iterates = newton_schulz_iterates(buf, max(NS_STEPS_TO_RECORD))
    s_norm = torch.linalg.svdvals(normalized.float())
    singular_values_by_stage["normalized"] = s_norm
    dist_norm = (normalized.float() - polar).norm().item() / max(polar.norm().item(), 1e-12)
    writer.writerow({"layer": name, "stage": "normalized", "dist_to_polar": dist_norm,
                      **spectrum_stats(s_norm)})

    for k in NS_STEPS_TO_RECORD:
        X_k = iterates[k - 1].float()
        s_k = torch.linalg.svdvals(X_k)
        singular_values_by_stage[f"ns_{k}"] = s_k
        dist = (X_k - polar).norm().item() / max(polar.norm().item(), 1e-12)
        writer.writerow({"layer": name, "stage": f"ns_{k}", "dist_to_polar": dist,
                          **spectrum_stats(s_k)})

    writer.writerow({"layer": name, "stage": "true_polar", "dist_to_polar": 0.0,
                      **spectrum_stats(s_polar)})

    return singular_values_by_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", choices=["early", "mid", "late", "all"], default="all",
        help="Which subset of checkpoints to process (for splitting into separate jobs).",
    )
    args = parser.parse_args()
    if args.group == "all":
        checkpoint_steps = [s for grp in CHECKPOINT_GROUPS.values() for s in grp]
    else:
        checkpoint_steps = CHECKPOINT_GROUPS[args.group]

    print(f"device={DEVICE}")
    print(f"group={args.group} -> checkpoint_steps={checkpoint_steps}")
    cfg, _ = utils.load_config(CONFIG_PATH, job_idx=None)
    model, _ = construct_model(cfg)
    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizer = intialize_optimizer(param_groups, cfg)
    matrix_params = {n: p for n, p in model.named_parameters() if p.ndim == 2}
    print(f"tracking {len(matrix_params)} matrix params (qkv split further at read time)")
    print(f"plotting histograms for: {sorted(PLOT_LAYERS)}")

    for step in checkpoint_steps:
        ckpt_out_dir = os.path.join(OUT_ROOT, f"ckpt_{step}")
        plot_dir = os.path.join(ckpt_out_dir, "singular_value_histograms")
        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(CKPT_DIR, f"ckpt_micro_step_{step}.pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer"])

        with open(os.path.join(ckpt_out_dir, "conditioning.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()

            for name, p in matrix_params.items():
                state = optimizer.state.get(p, {})
                if "q" in state and "k" in state and "v" in state:
                    subs = {"q": state["q"], "k": state["k"], "v": state["v"]}
                else:
                    subs = {"full": state}
                for sub_name, sub_state in subs.items():
                    buf = sub_state.get("momentum_buffer")
                    if buf is None:
                        continue
                    full_name = f"{name}.{sub_name}" if sub_name != "full" else name
                    svs = analyze_matrix(full_name, buf, writer)
                    if full_name in PLOT_LAYERS:
                        plot_histograms(
                            full_name, svs,
                            os.path.join(plot_dir, f"{full_name.replace('.', '_')}.pdf"),
                        )
            csvfile.flush()

        print(f"checkpoint {step}: done -> {ckpt_out_dir}")

    print("ALL DONE")


if __name__ == "__main__":
    main()
