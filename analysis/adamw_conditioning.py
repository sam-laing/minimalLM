"""AdamW analog of the momentum/gradient conditioning study: for the
adamw_90M_1p8B (no-rotation) baseline run, plots the normalized momentum
buffer (CustomAdamW's exp_avg - its first-moment/"momentum" state) next to
the normalized raw gradient, side by side, for every matrix.

Unlike Muon, AdamW has no matrix-structured orthogonalization step at all -
exp_avg is just an exponential moving average applied elementwise, with no
concept of Newton-Schulz iteration or a "true polar" target. So this script
only computes the "normalized" (Frobenius-norm-normalized) stage for each,
nothing else - there's no NS trajectory to show for an optimizer that never
treats the parameter as a matrix in the first place.

QKV is saved as a single fused (1728, 576) tensor for both exp_avg and the
gradient (AdamW, unlike Muon's sep_qkv, never splits it) - split manually
into q/k/v here purely for naming consistency with the Muon-side analysis,
so files line up layer-for-layer for comparison.

Output layout - same per-checkpoint directories as the Muon analyses:
  analysis/results/ckpt_<step>/adamw_normalized_conditioning.csv
  analysis/results/ckpt_<step>/singular_value_histograms/adamw_normalized_<layer>.pdf
    (2 panels: momentum | gradient)
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

CKPT_DIR = "/data/horse/ws/sala597i-slaing/checkpoints/adamw_90M_1p8B"
CONFIG_PATH = "config/config_adamw.yaml"
OUT_ROOT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_GROUPS = {
    "early": [471, 943, 1415],
    "mid": [13687, 14159, 14631],
    "late": [26903, 27375, 27452],
}

PLOT_SOURCES = ["momentum", "gradient"]

N_LAYERS = 8
PLOT_LAYERS = {"embed_tokens.weight", "lm_head.weight"}
for _l in range(N_LAYERS):
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_qkv.weight.q")
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_qkv.weight.k")
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_qkv.weight.v")
    PLOT_LAYERS.add(f"layers.{_l}.attn.w_out.weight")

FIELDNAMES = ["layer", "source", "sigma_max", "sigma_min", "sigma_mean",
              "sigma_std", "condition_number", "effective_rank"]


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


def normalized_svdvals(buf):
    X = buf.to(DEVICE).float()
    X = X / (X.norm() + 1e-7)
    return torch.linalg.svdvals(X)


def plot_side_by_side(name, s_by_source, out_path):
    fig, axes = plt.subplots(1, len(PLOT_SOURCES), figsize=(6, 3.5), sharey=False)
    for ax, source in zip(axes, PLOT_SOURCES):
        s = s_by_source[source]
        smin, smax = s.min().item(), s.max().item()
        condition_number = smax / max(smin, 1e-12)
        ax.hist(s.cpu().numpy(), bins=40, color="tab:green" if source == "momentum" else "tab:orange", alpha=0.8)
        ax.set_title(
            f"normalized {source}\ncond(max/min)={condition_number:.3g}\nmin={smin:.3g}  max={smax:.3g}",
            fontsize=8,
        )
        ax.set_xlabel("singular value")
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    axes[0].set_ylabel("count")
    fig.suptitle(f"AdamW: {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def analyze_matrix(name, momentum_buf, grad_buf, writer):
    s_by_source = {}
    for source, buf in [("momentum", momentum_buf), ("gradient", grad_buf)]:
        s = normalized_svdvals(buf)
        s_by_source[source] = s
        writer.writerow({"layer": name, "source": source, **spectrum_stats(s)})
    return s_by_source


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
    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizer = intialize_optimizer(param_groups, cfg)
    named_params = dict(model.named_parameters())
    matrix_params = {n: p for n, p in named_params.items() if p.ndim == 2}
    print(f"tracking {len(matrix_params)} matrix params (qkv split further at read time)")

    for step in checkpoint_steps:
        ckpt_out_dir = os.path.join(OUT_ROOT, f"ckpt_{step}")
        plot_dir = os.path.join(ckpt_out_dir, "singular_value_histograms")
        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(CKPT_DIR, f"ckpt_micro_step_{step}.pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        raw_grads = ckpt["grads"]
        if raw_grads is None:
            raise RuntimeError(f"checkpoint {step} has no saved grads")
        grads = {k.removeprefix("_orig_mod."): v for k, v in raw_grads.items()}

        with open(os.path.join(ckpt_out_dir, "adamw_normalized_conditioning.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()

            for name, p in matrix_params.items():
                state = optimizer.state.get(p, {})
                exp_avg = state.get("exp_avg")
                grad = grads.get(name)
                if exp_avg is None or grad is None:
                    continue

                if exp_avg.ndim == 2 and exp_avg.size(0) == 3 * exp_avg.size(1):
                    d = exp_avg.size(1)
                    m_q, m_k, m_v = exp_avg.split(d, dim=0)
                    g_q, g_k, g_v = grad.split(d, dim=0)
                    subs = {"q": (m_q, g_q), "k": (m_k, g_k), "v": (m_v, g_v)}
                else:
                    subs = {"full": (exp_avg, grad)}

                for sub_name, (m_buf, g_buf) in subs.items():
                    full_name = f"{name}.{sub_name}" if sub_name != "full" else name
                    s_by_source = analyze_matrix(full_name, m_buf, g_buf, writer)
                    if full_name in PLOT_LAYERS:
                        plot_side_by_side(
                            full_name, s_by_source,
                            os.path.join(plot_dir, f"adamw_normalized_{full_name.replace('.', '_')}.pdf"),
                        )
            csvfile.flush()

        print(f"checkpoint {step}: done -> {ckpt_out_dir}")

    print("ALL DONE")


if __name__ == "__main__":
    main()
