import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from plot_svs_per_head import load_checkpoint, compute_svs, split_qkv_per_head


def get_out_svs_per_head(state_dict, layer_idx, num_heads):
    """Get singular values for W_out per head: W_out is [d_model, d_model], split cols into heads."""
    out_key = f'layers.{layer_idx}.attn.w_out.weight'
    w_out = state_dict[out_key]  # [d_model, d_model]
    d = w_out.shape[0]
    head_dim = d // num_heads
    # Split along input dim (columns) so each head block is [d_model, head_dim]
    out_svs = [compute_svs(w_out[:, h * head_dim:(h + 1) * head_dim]) for h in range(num_heads)]
    return out_svs


def plot_per_layer_svd():
    """For each checkpoint step and layer, create a 2x4 plot with per-head SV curves."""
    device = torch.device("cpu")

    checkpoint_steps = [1550, 3100, 4650, 6200]
    num_layers = 12
    num_heads = 12
    matrix_names = ['Q', 'K', 'V', 'W_out']

    colors_list = ['#FFD700', '#9ACD32', '#228B22', '#20B2AA', '#4169E1', '#000080']
    head_cmap = LinearSegmentedColormap.from_list('yellow_green_blue', colors_list)

    base_dir = 'weight_analysis/svd_per_layer'

    for step in checkpoint_steps:
        step_dir = os.path.join(base_dir, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)

        adam_path = f"/home/fast/slaing/plainLMcheckpoints/custom_adamw/ckpt_micro_step_{step}.pth"
        muon_path = f"/home/fast/slaing/plainLMcheckpoints/muon/ckpt_micro_step_{step}.pth"
        print(f"\nLoading checkpoints for step {step}...")
        adam_state = load_checkpoint(adam_path, device)
        muon_state = load_checkpoint(muon_path, device)

        for layer_idx in range(num_layers):
            fig, axes = plt.subplots(2, 4, figsize=(22, 8), sharey='row')

            for row, (opt_name, state_dict) in enumerate([('Adam', adam_state), ('Muon', muon_state)]):
                qkv_key = f'layers.{layer_idx}.attn.w_qkv.weight'
                w_qkv = state_dict[qkv_key]
                w_q, w_k, w_v = split_qkv_per_head(w_qkv, num_heads)

                q_svs = [compute_svs(w_q[h]) for h in range(num_heads)]
                k_svs = [compute_svs(w_k[h]) for h in range(num_heads)]
                v_svs = [compute_svs(w_v[h]) for h in range(num_heads)]
                out_svs = get_out_svs_per_head(state_dict, layer_idx, num_heads)

                all_svs = [q_svs, k_svs, v_svs, out_svs]

                for col, (mat_name, svs_list) in enumerate(zip(matrix_names, all_svs)):
                    ax = axes[row, col]
                    for head_idx, sv in enumerate(svs_list):
                        color = head_cmap(head_idx / (num_heads - 1))
                        ax.semilogy(sv, color=color, linewidth=2.0, alpha=0.8)

                    ax.set_title(f'{opt_name} - {mat_name}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('SV Index', fontsize=10)
                    if col == 0:
                        ax.set_ylabel('Singular Value (log)', fontsize=10)
                    ax.grid(True, alpha=0.3)

            fig.suptitle(
                f'Layer {layer_idx}: Per-Head SV Structure — Adam vs Muon (Step {step})\n'
                f'Color: Head Index (Yellow=H0, Blue=H{num_heads - 1})',
                fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 0.92, 0.93])

            cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
            sm = plt.cm.ScalarMappable(cmap=head_cmap, norm=plt.Normalize(vmin=0, vmax=num_heads - 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Head', fontsize=11)
            cbar.set_ticks([0, num_heads // 2, num_heads - 1])
            cbar.set_ticklabels(['H0', f'H{num_heads // 2}', f'H{num_heads - 1}'])

            save_path = os.path.join(step_dir, f'layer_{layer_idx}_svd.pdf')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")

        del adam_state, muon_state

    print(f"\nAll plots saved under {base_dir}/")


if __name__ == "__main__":
    plot_per_layer_svd()
