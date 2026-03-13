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
    
    # Yellow -> Green -> Blue colormap for heads
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#FFD700', '#9ACD32', '#228B22', '#20B2AA', '#4169E1', '#000080']
    head_cmap = LinearSegmentedColormap.from_list('yellow_green_blue', colors_list)
    
    base_dir = 'weight_analysis/svd_per_head'
    
    for step in checkpoint_steps:
        step_dir = os.path.join(base_dir, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)

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
        
        for layer_idx in range(num_layers):
            # 2 rows (Adam, Muon) x 3 cols (Q, K, V)
            fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey='row')
            
            for row, (opt_name, state_dict) in enumerate([('Adam', adam_state), ('Muon', muon_state)]):
                q_svs, k_svs, v_svs = get_qkv_svs_per_head(state_dict, layer_idx, num_heads)
                
                for col, (mat_name, svs_list) in enumerate([('Q', q_svs), ('K', k_svs), ('V', v_svs)]):
                    ax = axes[row, col]
                    
                    for head_idx, sv in enumerate(svs_list):
                        color = head_cmap(head_idx / (num_heads - 1))
                        ax.semilogy(sv, color=color, linewidth=2.0, alpha=0.8, label=f'H{head_idx}')
                    
                    ax.set_title(f'{opt_name} - {mat_name}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('SV Index', fontsize=9)
                    if col == 0:
                        ax.set_ylabel('Singular Value (log)', fontsize=10)
                    ax.grid(True, alpha=0.3)
        
            fig.suptitle(f'Layer {layer_idx}: Per-Head SV Structure — Adam vs Muon (Step {step})\n'
                         f'Color: Head Index (Yellow=H0, Blue=H{num_heads-1})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 0.92, 0.93])
        
            # Add colorbar for heads
            cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
            sm = plt.cm.ScalarMappable(cmap=head_cmap, norm=plt.Normalize(vmin=0, vmax=num_heads-1))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Head', fontsize=11)
            cbar.set_ticks([0, num_heads//2, num_heads-1])
            cbar.set_ticklabels(['H0', f'H{num_heads//2}', f'H{num_heads-1}'])
        
            save_path = os.path.join(step_dir, f'layer_{layer_idx}_per_head_svd.pdf')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")

        del adam_state, muon_state
    
    print("\n" + "="*60)
    print("All per-head plots generated!")
    print("="*60)


if __name__ == "__main__":
    compare_adam_muon_svd_per_head()
