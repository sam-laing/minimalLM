import torch 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def compute_svs(weight_tensor):
    """Compute singular values of a weight matrix."""
    W = weight_tensor.float().cpu()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    return S.numpy()

def split_qkv(w_qkv):
    """Split fused QKV weight [3*d, d] into Q, K, V each [d, d]."""
    d = w_qkv.shape[0] // 3
    w_q = w_qkv[:d, :]
    w_k = w_qkv[d:2*d, :]
    w_v = w_qkv[2*d:, :]
    return w_q, w_k, w_v

def load_checkpoint(path, device):
    """Load checkpoint and return state dict."""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint['state_dict']

def get_qkv_svs(state_dict, layer_idx):
    """Get singular values for Q, K, V matrices at a given layer."""
    qkv_key = f'layers.{layer_idx}.attn.w_qkv.weight'
    w_qkv = state_dict[qkv_key]
    w_q, w_k, w_v = split_qkv(w_qkv)
    return compute_svs(w_q), compute_svs(w_k), compute_svs(w_v)

def get_out_svs(state_dict, layer_idx):
    """Get singular values for output projection matrix at a given layer."""
    out_key = f'layers.{layer_idx}.attn.w_out.weight'
    w_out = state_dict[out_key]
    return compute_svs(w_out)

def get_embedding_svs(state_dict):
    """Get singular values for the token embedding matrix."""
    embed_key = 'embed_tokens.weight'
    w_embed = state_dict[embed_key]
    return compute_svs(w_embed)

def export_svs_to_csv(state_dict, optimizer_name, step, num_layers, output_dir):
    """Export singular values to CSV files for a given checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all layer SVs for Q, K, V, W_out
    all_q_svs, all_k_svs, all_v_svs, all_out_svs = [], [], [], []
    for layer_idx in range(num_layers):
        q_sv, k_sv, v_sv = get_qkv_svs(state_dict, layer_idx)
        out_sv = get_out_svs(state_dict, layer_idx)
        all_q_svs.append(q_sv)
        all_k_svs.append(k_sv)
        all_v_svs.append(v_sv)
        all_out_svs.append(out_sv)
    
    # Create DataFrames with layers as columns
    matrix_data = {
        'Q': all_q_svs,
        'K': all_k_svs,
        'V': all_v_svs,
        'W_out': all_out_svs,
    }
    
    for mat_name, svs_list in matrix_data.items():
        # Each column is a layer, each row is a singular value index
        df = pd.DataFrame({f'layer_{i}': svs for i, svs in enumerate(svs_list)})
        df.index.name = 'sv_index'
        csv_path = os.path.join(output_dir, f'{optimizer_name}_{mat_name}_step{step}.csv')
        df.to_csv(csv_path)
        print(f"Saved: {csv_path}")
    
    # Embedding (single vector)
    embed_sv = get_embedding_svs(state_dict)
    embed_df = pd.DataFrame({'singular_values': embed_sv})
    embed_df.index.name = 'sv_index'
    embed_csv_path = os.path.join(output_dir, f'{optimizer_name}_embedding_step{step}.csv')
    embed_df.to_csv(embed_csv_path)
    print(f"Saved: {embed_csv_path}")

def compare_adam_muon_svd():
    """Compare singular value distributions between Adam and Muon optimizers."""
    device = torch.device("cpu")
    
    # Checkpoint steps to analyze
    checkpoint_steps = [1550, 3100, 4650, 6200]
    
    # All layers (adjust num_layers based on your model)
    num_layers = 12
    layer_indices = list(range(num_layers))
    matrix_types = ['Query (Q)', 'Key (K)', 'Value (V)', 'Output (W_out)', 'Embedding']
    
    # Yellow -> Green -> Blue colormap (warm early, cool late)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#FFD700', '#9ACD32', '#228B22', '#20B2AA', '#4169E1', '#000080']  # Gold -> YellowGreen -> ForestGreen -> LightSeaGreen -> RoyalBlue -> Navy
    layer_cmap = LinearSegmentedColormap.from_list('yellow_green_blue', colors_list)
    
    os.makedirs('model_comparison_plots/svd', exist_ok=True)
    
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
        
        # Export SVs to CSV for the last checkpoint
        if step == checkpoint_steps[-1]:
            csv_output_dir = 'model_comparison_plots/svd/csv'
            print(f"\nExporting singular values to CSV for step {step}...")
            export_svs_to_csv(adam_state, 'adam', step, num_layers, csv_output_dir)
            export_svs_to_csv(muon_state, 'muon', step, num_layers, csv_output_dir)
        
        # Create figure: 2 rows x 5 cols (Q, K, V, W_out, Embedding)
        # Share y-axis within each row for easier comparison
        fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey='row')
        
        optimizer_data = [
            ('Adam', adam_state, axes[0, :]),
            ('Muon', muon_state, axes[1, :])
        ]
        
        for opt_name, state_dict, ax_row in optimizer_data:
            # Collect SVs for all layers
            all_q_svs, all_k_svs, all_v_svs, all_out_svs = [], [], [], []
            for layer_idx in layer_indices:
                q_sv, k_sv, v_sv = get_qkv_svs(state_dict, layer_idx)
                out_sv = get_out_svs(state_dict, layer_idx)
                all_q_svs.append(q_sv)
                all_k_svs.append(k_sv)
                all_v_svs.append(v_sv)
                all_out_svs.append(out_sv)
            
            # Get embedding SVs (single matrix, not per-layer)
            embed_sv = get_embedding_svs(state_dict)
            
            all_svs = [all_q_svs, all_k_svs, all_v_svs, all_out_svs]
            
            for col, (svs_list, mat_name) in enumerate(zip(all_svs, matrix_types[:-1])):
                ax = ax_row[col]
                
                for layer_idx, sv in enumerate(svs_list):
                    # Color from yellow (early) to blue (late) based on layer depth
                    color = layer_cmap(layer_idx / (num_layers - 1))
                    ax.semilogy(sv, color=color, linewidth=2.5, alpha=0.85, label=f'L{layer_idx}')
                
                ax.set_title(f'{opt_name} - {mat_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Singular Value Index', fontsize=10)
                if col == 0:
                    ax.set_ylabel('Singular Value (log)', fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Plot embedding in the 5th column
            ax_embed = ax_row[4]
            ax_embed.semilogy(embed_sv, color='#8B008B', linewidth=2.5, alpha=0.85)  # Dark magenta
            ax_embed.set_title(f'{opt_name} - Embedding', fontsize=12, fontweight='bold')
            ax_embed.set_xlabel('Singular Value Index', fontsize=10)
            ax_embed.grid(True, alpha=0.3)
        
        plt.suptitle(f'Adam vs Muon: Singular Value Distributions (Step {step})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.90, 0.96])
        
        # Add colorbar to the right, outside the plots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=layer_cmap, norm=plt.Normalize(vmin=0, vmax=num_layers-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Layer', fontsize=11)
        cbar.set_ticks([0, num_layers//2, num_layers-1])
        cbar.set_ticklabels(['L0\n(early)', f'L{num_layers//2}\n(mid)', f'L{num_layers-1}\n(late)'])
        
        # Save figure
        plt.savefig(f'model_comparison_plots/svd/adam_vs_muon_qkv_svd_step{step}.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: model_comparison_plots/svd/adam_vs_muon_qkv_svd_step{step}.pdf")
    
    print("\n" + "="*60)
    print("All plots generated!")
    print("="*60)


if __name__ == "__main__":
    compare_adam_muon_svd() 
