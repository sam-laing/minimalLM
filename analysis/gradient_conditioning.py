"""Same conditioning study as momentum_conditioning.py, but for the raw
gradient at each checkpoint (ckpt['grads'], stashed by engine.py right
before zero_grad() - see checkpoint_utils.py) instead of Muon's momentum
buffer. Output filenames are prefixed with grad_ and land in the *same*
per-checkpoint directories as the momentum-buffer results, so the two are
directly side by side for comparison.

QKV gradients are saved as a single fused (1728, 576) tensor (unlike the
momentum buffer, which Muon's sep_qkv already stores split into three
separate (576, 576) buffers) - so here we manually split the raw gradient
into q/k/v the same way Muon's _muon_step does (g.split(d, dim=0)), to stay
directly comparable layer-for-layer with the momentum-buffer analysis.

For every matrix (QKV split into q/k/v), computes singular-value-based
conditioning of:
  - "raw":       the gradient itself, before any orthogonalization
  - "normalized": the gradient after Frobenius-norm normalization (the
                  actual starting point of Muon's NS iteration - see
                  momentum_conditioning.py for why this isn't max-sv-1)
  - "ns_1".."ns_5": the Newton-Schulz iterate after k steps, replicating
                  optim/muon.py's zeropower_via_newtonschulz5 exactly
                  (verified bit-for-bit against the real function)
  - "true_polar": the exact polar factor (full SVD, U @ Vh)

Output layout - one directory per checkpoint, same as momentum_conditioning.py:
  analysis/results/ckpt_<step>/grad_conditioning.csv
  analysis/results/ckpt_<step>/singular_value_histograms/grad_<layer>.pdf
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import utils
from models import construct_model

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
PLOT_STAGES = ["normalized"] + [f"ns_{k}" for k in NS_STEPS_TO_RECORD]

# exact constants from optim/muon.py's zeropower_via_newtonschulz5 - verified
# bit-for-bit against the current source
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_EPS = 1e-7

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
    """Bit-for-bit replica of optim/muon.py's zeropower_via_newtonschulz5
    (verified against the real function), returning the normalized
    starting point plus the iterate after every step."""
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
        condition_number = smax / max(smin, 1e-12)
        ax.hist(s.cpu().numpy(), bins=40, color="tab:orange", alpha=0.8)
        ax.set_title(
            f"grad {stage}\ncond(max/min)={condition_number:.3g}\nmin={smin:.3g}  max={smax:.3g}",
            fontsize=8,
        )
        ax.set_xlabel("singular value")
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    axes[0].set_ylabel("count")
    fig.suptitle(f"grad: {name}")
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
    parser.add_argument("--group", choices=["early", "mid", "late", "all"], default="all")
    args = parser.parse_args()
    if args.group == "all":
        checkpoint_steps = [s for grp in CHECKPOINT_GROUPS.values() for s in grp]
    else:
        checkpoint_steps = CHECKPOINT_GROUPS[args.group]

    print(f"device={DEVICE}")
    print(f"group={args.group} -> checkpoint_steps={checkpoint_steps}")
    cfg, _ = utils.load_config(CONFIG_PATH, job_idx=None)
    model, _ = construct_model(cfg)
    matrix_params = [n for n, p in model.named_parameters() if p.ndim == 2]
    print(f"tracking {len(matrix_params)} matrix params (qkv split further at read time)")
    print(f"plotting histograms for: {sorted(PLOT_LAYERS)}")

    for step in checkpoint_steps:
        ckpt_out_dir = os.path.join(OUT_ROOT, f"ckpt_{step}")
        plot_dir = os.path.join(ckpt_out_dir, "singular_value_histograms")
        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(CKPT_DIR, f"ckpt_micro_step_{step}.pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw_grads = ckpt["grads"]
        if raw_grads is None:
            raise RuntimeError(f"checkpoint {step} has no saved grads")
        # torch.compile wraps the model, so saved grad keys are prefixed
        # with "_orig_mod." - strip it to match model.named_parameters()
        grads = {k.removeprefix("_orig_mod."): v for k, v in raw_grads.items()}

        with open(os.path.join(ckpt_out_dir, "grad_conditioning.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()

            for name in matrix_params:
                g = grads.get(name)
                if g is None:
                    continue
                if g.ndim == 2 and g.size(0) == 3 * g.size(1):
                    d = g.size(1)
                    g_q, g_k, g_v = g.split(d, dim=0)
                    subs = {"q": g_q, "k": g_k, "v": g_v}
                else:
                    subs = {"full": g}
                for sub_name, buf in subs.items():
                    full_name = f"{name}.{sub_name}" if sub_name != "full" else name
                    svs = analyze_matrix(full_name, buf, writer)
                    if full_name in PLOT_LAYERS:
                        plot_histograms(
                            full_name, svs,
                            os.path.join(plot_dir, f"grad_{full_name.replace('.', '_')}.pdf"),
                        )
            csvfile.flush()

        print(f"checkpoint {step}: done -> {ckpt_out_dir}")

    print("ALL DONE")


if __name__ == "__main__":
    main()
