import torch 
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_svs(weight_tensor):
    """Compute singular values of a weight matrix."""
    W = weight_tensor.float().cpu()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    return S.numpy()

def split_qkv_per_head(w_qkv, num_heads):
    """Split fused QKV weight into per-head matrices.
    
    Args:
        w_qkv: [3*d_model, d_model] fused QKV weight
        num_heads: number of attention heads
    
    Returns:
        w_q, w_k, w_v: each [num_heads, head_dim, d_model]
    """
    d = w_qkv.shape[0] // 3
    head_dim = d // num_heads
    
    w_q = w_qkv[:d, :].reshape(num_heads, head_dim, -1)
    w_k = w_qkv[d:2*d, :].reshape(num_heads, head_dim, -1)
    w_v = w_qkv[2*d:, :].reshape(num_heads, head_dim, -1)
    
    return w_q, w_k, w_v

def load_checkpoint(path, device):
    """Load checkpoint and return state dict."""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint['state_dict']

def get_qkv_svs_per_head(state_dict, layer_idx, num_heads):
    """Get singular values for Q, K, V matrices per head at a given layer.
    
    Returns:
        List of (q_svs, k_svs, v_svs) where each is [num_heads] arrays of SVs
    """
    qkv_key = f'layers.{layer_idx}.attn.w_qkv.weight'
    w_qkv = state_dict[qkv_key]
    w_q, w_k, w_v = split_qkv_per_head(w_qkv, num_heads)
    
    q_svs = [compute_svs(w_q[h]) for h in range(num_heads)]
    k_svs = [compute_svs(w_k[h]) for h in range(num_heads)]
    v_svs = [compute_svs(w_v[h]) for h in range(num_heads)]
    
    return q_svs, k_svs, v_svs

def compare_adam_muon_svd_per_head():
    """Compare per-head singular value distributions between Adam and Muon optimizers."""
    device = torch.device("cpu")
    
    # Checkpoint steps to analyze
    checkpoint_steps = [1550, 3100, 4650, 6200]
    
    # Model config
    num_layers = 12
    num_heads = 12  # Adjust based on your model
    
    # Select representative layers: beginning, middle, end
    select_layers = [0, 6, 11]
    layer_names = ['Layer 0 (Early)', 'Layer 6 (Mid)', 'Layer 11 (Late)']
    
    # Yellow -> Green -> Blue colormap for heads
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#FFD700', '#9ACD32', '#228B22', '#20B2AA', '#4169E1', '#000080']
    head_cmap = LinearSegmentedColormap.from_list('yellow_green_blue', colors_list)
    
    os.makedirs('model_comparison_plots/svd_sep', exist_ok=True)
    
    for step in checkpoint_steps:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint step {step}")
        print('='*60)
        
        # Checkpoint paths
        adam_path = f"/home/fast/slaing/plainLMcheckpoints/custom_adamw/ckpt_micro_step_{step}.pth"
        muon_path = f"/home/fast/slaing/plainLMcheckpoints/muon/ckpt_micro_step_{step}.pth"
        
        # Load checkpoints
        print("Loading Adam checkpoint...")
        adam_state = load_checkpoint(adam_path, device)
        print("Loading Muon checkpoint...")
        muon_state = load_checkpoint(muon_path, device)
        
        # Create figure: 2 rows (Adam, Muon) x 9 cols (3 layers x 3 matrix types Q/K/V)
        fig, axes = plt.subplots(2, 9, figsize=(28, 8), sharey='row')
        
        optimizer_data = [
            ('Adam', adam_state, axes[0, :]),
            ('Muon', muon_state, axes[1, :])
        ]
        
        for opt_name, state_dict, ax_row in optimizer_data:
            col = 0
            for layer_idx, layer_name in zip(select_layers, layer_names):
                q_svs, k_svs, v_svs = get_qkv_svs_per_head(state_dict, layer_idx, num_heads)
                
                for mat_name, svs_list in [('Q', q_svs), ('K', k_svs), ('V', v_svs)]:
                    ax = ax_row[col]
                    
                    for head_idx, sv in enumerate(svs_list):
                        color = head_cmap(head_idx / (num_heads - 1))
                        ax.semilogy(sv, color=color, linewidth=2.0, alpha=0.8, label=f'H{head_idx}')
                    
                    ax.set_title(f'{opt_name} L{layer_idx} {mat_name}', fontsize=10, fontweight='bold')
                    ax.set_xlabel('SV Index', fontsize=9)
                    if col == 0:
                        ax.set_ylabel('Singular Value (log)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    col += 1
        
        plt.suptitle(f'Adam vs Muon: Per-Head Singular Value Distributions (Step {step})\nColor: Head Index (Yellow=H0, Blue=H{num_heads-1})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.92, 0.94])
        
        # Add colorbar for heads
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        sm = plt.cm.ScalarMappable(cmap=head_cmap, norm=plt.Normalize(vmin=0, vmax=num_heads-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Head', fontsize=11)
        cbar.set_ticks([0, num_heads//2, num_heads-1])
        cbar.set_ticklabels(['H0', f'H{num_heads//2}', f'H{num_heads-1}'])
        
        # Save figure
        plt.savefig(f'model_comparison_plots/svd_sep/adam_vs_muon_per_head_step{step}.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: model_comparison_plots/svd_sep/adam_vs_muon_per_head_step{step}.pdf")
    
    print("\n" + "="*60)
    print("All per-head plots generated!")
    print("="*60)


if __name__ == "__main__":
    compare_adam_muon_svd_per_head()
