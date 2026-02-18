import torch
import time
import os 
from dataclasses import dataclass, field
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

@dataclass
class Config:
    """Dynamic config dataclass that accepts any fields from yaml."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'Config({attrs})'

def load_model(model_path, device):
    """Load a model from checkpoint."""
    cfg_path = os.path.join(model_path, "config.yaml")
    import yaml
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = Config(**cfg_dict)
    
    from models import construct_model
    model, model_cfg = construct_model(cfg)
    
    # Find checkpoint file (check both .pt and .pth extensions)
    ckpt_files = [f for f in os.listdir(model_path) if f.startswith("ckpt_") and (f.endswith(".pt") or f.endswith(".pth"))]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {model_path}")
    weights_path = os.path.join(model_path, ckpt_files[0])
    print(f"Loading checkpoint: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model, cfg

def get_linear_layers(model):
    """Extract all linear layer weights with their names."""
    linear_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_weights[name] = module.weight.detach().cpu()
    return linear_weights

def compute_singular_values(weight_matrix):
    """Compute singular values of a weight matrix."""
    U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
    return S.numpy()

def plot_singular_values_comparison(weights1, weights2, name1="Model 1", name2="Model 2", save_dir="./svd_plots"):
    """Plot singular value comparison for each linear layer."""
    os.makedirs(save_dir, exist_ok=True)
    
    common_layers = set(weights1.keys()) & set(weights2.keys())
    
    for layer_name in sorted(common_layers):
        w1, w2 = weights1[layer_name], weights2[layer_name]
        
        sv1 = compute_singular_values(w1)
        sv2 = compute_singular_values(w2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Singular values (log scale)
        axes[0].semilogy(sv1, label=name1, alpha=0.8)
        axes[0].semilogy(sv2, label=name2, alpha=0.8)
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Singular Value (log)")
        axes[0].set_title(f"Singular Values: {layer_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Normalized singular values (spectrum shape)
        axes[1].plot(sv1 / sv1[0], label=name1, alpha=0.8)
        axes[1].plot(sv2 / sv2[0], label=name2, alpha=0.8)
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Normalized SV (σ_i / σ_0)")
        axes[1].set_title(f"Normalized Spectrum: {layer_name}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative energy
        energy1 = np.cumsum(sv1**2) / np.sum(sv1**2)
        energy2 = np.cumsum(sv2**2) / np.sum(sv2**2)
        axes[2].plot(energy1, label=name1, alpha=0.8)
        axes[2].plot(energy2, label=name2, alpha=0.8)
        axes[2].set_xlabel("Index")
        axes[2].set_ylabel("Cumulative Energy")
        axes[2].set_title(f"Cumulative Energy: {layer_name}")
        axes[2].axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='99% energy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = layer_name.replace(".", "_")
        plt.savefig(os.path.join(save_dir, f"svd_{safe_name}.png"), dpi=150)
        plt.close()
        
        # Print statistics
        eff_rank1 = np.sum(sv1 > sv1[0] * 1e-3)
        eff_rank2 = np.sum(sv2 > sv2[0] * 1e-3)
        condition1 = sv1[0] / sv1[-1] if sv1[-1] > 0 else float('inf')
        condition2 = sv2[0] / sv2[-1] if sv2[-1] > 0 else float('inf')
        
        print(f"{layer_name}:")
        print(f"  Shape: {tuple(w1.shape)}")
        print(f"  {name1} - σ_max: {sv1[0]:.4f}, σ_min: {sv1[-1]:.6f}, cond: {condition1:.2f}, eff_rank: {eff_rank1}")
        print(f"  {name2} - σ_max: {sv2[0]:.4f}, σ_min: {sv2[-1]:.6f}, cond: {condition2:.2f}, eff_rank: {eff_rank2}")
        print()

def plot_weight_statistics(weights1, weights2, name1="Model 1", name2="Model 2", save_dir="./weight_plots"):
    """Plot weight distribution statistics."""
    os.makedirs(save_dir, exist_ok=True)
    
    common_layers = set(weights1.keys()) & set(weights2.keys())
    
    stats1 = {"mean": [], "std": [], "max": [], "min": [], "names": []}
    stats2 = {"mean": [], "std": [], "max": [], "min": [], "names": []}
    
    for layer_name in sorted(common_layers):
        w1, w2 = weights1[layer_name].numpy(), weights2[layer_name].numpy()
        
        stats1["mean"].append(np.mean(w1))
        stats1["std"].append(np.std(w1))
        stats1["max"].append(np.max(np.abs(w1)))
        stats1["min"].append(np.min(np.abs(w1)))
        stats1["names"].append(layer_name)
        
        stats2["mean"].append(np.mean(w2))
        stats2["std"].append(np.std(w2))
        stats2["max"].append(np.max(np.abs(w2)))
        stats2["min"].append(np.min(np.abs(w2)))
        stats2["names"].append(layer_name)
    
    # Plot weight std per layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(stats1["names"]))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, stats1["std"], width, label=name1, alpha=0.8)
    axes[0, 0].bar(x + width/2, stats2["std"], width, label=name2, alpha=0.8)
    axes[0, 0].set_ylabel("Std Dev")
    axes[0, 0].set_title("Weight Standard Deviation per Layer")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(stats1["names"], rotation=90, fontsize=6)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(x - width/2, stats1["max"], width, label=name1, alpha=0.8)
    axes[0, 1].bar(x + width/2, stats2["max"], width, label=name2, alpha=0.8)
    axes[0, 1].set_ylabel("Max |weight|")
    axes[0, 1].set_title("Max Absolute Weight per Layer")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(stats1["names"], rotation=90, fontsize=6)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of all weights
    all_w1 = np.concatenate([weights1[n].numpy().flatten() for n in sorted(common_layers)])
    all_w2 = np.concatenate([weights2[n].numpy().flatten() for n in sorted(common_layers)])
    
    axes[1, 0].hist(all_w1, bins=100, alpha=0.6, label=name1, density=True)
    axes[1, 0].hist(all_w2, bins=100, alpha=0.6, label=name2, density=True)
    axes[1, 0].set_xlabel("Weight Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Weight Distribution (All Layers)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frobenius norm per layer
    frob1 = [np.linalg.norm(weights1[n].numpy()) for n in sorted(common_layers)]
    frob2 = [np.linalg.norm(weights2[n].numpy()) for n in sorted(common_layers)]
    
    axes[1, 1].bar(x - width/2, frob1, width, label=name1, alpha=0.8)
    axes[1, 1].bar(x + width/2, frob2, width, label=name2, alpha=0.8)
    axes[1, 1].set_ylabel("Frobenius Norm")
    axes[1, 1].set_title("Weight Frobenius Norm per Layer")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stats1["names"], rotation=90, fontsize=6)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_statistics.png"), dpi=150)
    plt.close()
    
    print(f"Weight statistics plots saved to {save_dir}")

def plot_weight_heatmaps(weights1, weights2, name1="Model 1", name2="Model 2", save_dir="./heatmap_plots"):
    """Plot weight heatmaps for select layers."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Pick a few representative layers
    common_layers = sorted(set(weights1.keys()) & set(weights2.keys()))
    select_layers = [l for l in common_layers if "w_qkv" in l or "w_out" in l or "fc1" in l][:4]
    
    for layer_name in select_layers:
        w1, w2 = weights1[layer_name].numpy(), weights2[layer_name].numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        vmax = max(np.abs(w1).max(), np.abs(w2).max())
        
        im1 = axes[0].imshow(w1, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"{name1}\n{layer_name}")
        axes[0].set_xlabel("Input dim")
        axes[0].set_ylabel("Output dim")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(w2, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"{name2}\n{layer_name}")
        axes[1].set_xlabel("Input dim")
        axes[1].set_ylabel("Output dim")
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = w1 - w2
        im3 = axes[2].imshow(diff, aspect='auto', cmap='coolwarm')
        axes[2].set_title(f"Difference\n{layer_name}")
        axes[2].set_xlabel("Input dim")
        axes[2].set_ylabel("Output dim")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        safe_name = layer_name.replace(".", "_")
        plt.savefig(os.path.join(save_dir, f"heatmap_{safe_name}.png"), dpi=150)
        plt.close()


# ============================================================================
# SINGLE MODEL ANALYSIS - Extract generalizations about converged weights
# ============================================================================

def power_law(x, a, b):
    """Power law function: f(x) = a * x^(-b)"""
    return a * np.power(x, -b)

def analyze_singular_value_decay(weights, save_dir="./analysis"):
    """Analyze singular value decay patterns across all layers."""
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        "layer_name": [],
        "shape": [],
        "sigma_max": [],
        "sigma_min": [],
        "condition_number": [],
        "effective_rank_99": [],  # rank to capture 99% energy
        "effective_rank_ratio": [],  # effective_rank / min(m,n)
        "power_law_exponent": [],  # decay rate
        "stable_rank": [],  # ||W||_F^2 / ||W||_2^2
    }
    
    all_normalized_svs = []
    layer_types = {"qkv": [], "out": [], "fc1": [], "fc2": [], "other": []}
    
    for layer_name in sorted(weights.keys()):
        w = weights[layer_name].numpy()
        sv = compute_singular_values(weights[layer_name])
        
        # Basic stats
        results["layer_name"].append(layer_name)
        results["shape"].append(w.shape)
        results["sigma_max"].append(sv[0])
        results["sigma_min"].append(sv[-1] if sv[-1] > 1e-10 else 1e-10)
        results["condition_number"].append(sv[0] / max(sv[-1], 1e-10))
        
        # Effective rank (99% energy)
        energy = np.cumsum(sv**2) / np.sum(sv**2)
        eff_rank_99 = np.searchsorted(energy, 0.99) + 1
        results["effective_rank_99"].append(eff_rank_99)
        results["effective_rank_ratio"].append(eff_rank_99 / len(sv))
        
        # Stable rank: ||W||_F^2 / ||W||_2^2
        stable_rank = np.sum(sv**2) / (sv[0]**2)
        results["stable_rank"].append(stable_rank)
        
        # Fit power law to singular value decay
        try:
            x_data = np.arange(1, len(sv) + 1)
            popt, _ = curve_fit(power_law, x_data[:len(sv)//2], sv[:len(sv)//2], 
                               p0=[sv[0], 0.5], maxfev=5000)
            results["power_law_exponent"].append(popt[1])
        except:
            results["power_law_exponent"].append(np.nan)
        
        # Normalized singular values for aggregation
        normalized_sv = sv / sv[0]
        all_normalized_svs.append(normalized_sv)
        
        # Categorize by layer type
        if "qkv" in layer_name.lower() or "w_q" in layer_name or "w_k" in layer_name or "w_v" in layer_name:
            layer_types["qkv"].append((layer_name, sv))
        elif "out" in layer_name.lower() or "w_o" in layer_name:
            layer_types["out"].append((layer_name, sv))
        elif "fc1" in layer_name or "up" in layer_name or "gate" in layer_name:
            layer_types["fc1"].append((layer_name, sv))
        elif "fc2" in layer_name or "down" in layer_name:
            layer_types["fc2"].append((layer_name, sv))
        else:
            layer_types["other"].append((layer_name, sv))
    
    return results, all_normalized_svs, layer_types

def analyze_weight_distributions(weights, save_dir="./analysis"):
    """Analyze weight distribution characteristics."""
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        "layer_name": [],
        "mean": [],
        "std": [],
        "skewness": [],
        "kurtosis": [],
        "sparsity_01": [],  # % weights < 0.01 * std
        "sparsity_001": [],  # % weights < 0.001 * std
        "max_abs": [],
        "l1_norm": [],
        "l2_norm": [],
        "linf_norm": [],
    }
    
    for layer_name in sorted(weights.keys()):
        w = weights[layer_name].numpy().flatten()
        
        results["layer_name"].append(layer_name)
        results["mean"].append(np.mean(w))
        results["std"].append(np.std(w))
        results["skewness"].append(stats.skew(w))
        results["kurtosis"].append(stats.kurtosis(w))
        
        # Sparsity measures
        threshold_01 = 0.01 * np.std(w)
        threshold_001 = 0.001 * np.std(w)
        results["sparsity_01"].append(np.mean(np.abs(w) < threshold_01) * 100)
        results["sparsity_001"].append(np.mean(np.abs(w) < threshold_001) * 100)
        
        # Norms
        results["max_abs"].append(np.max(np.abs(w)))
        results["l1_norm"].append(np.sum(np.abs(w)))
        results["l2_norm"].append(np.linalg.norm(w))
        results["linf_norm"].append(np.max(np.abs(w)))
    
    return results

def generate_weight_report(weights, model_name, save_dir="./weight_analysis"):
    """Generate comprehensive report on converged weight structure using sample layers."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"CONVERGED WEIGHT ANALYSIS: {model_name}")
    print("="*70)
    
    # =========================================================================
    # SELECT SAMPLE LAYERS (beginning, middle, end)
    # =========================================================================
    all_layer_names = sorted(weights.keys())
    
    # Extract layer indices and group by depth
    layer_by_depth = {}
    for name in all_layer_names:
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                depth = int(p)
                if depth not in layer_by_depth:
                    layer_by_depth[depth] = []
                layer_by_depth[depth].append(name)
                break
    
    # Select layers from beginning, middle, end
    depths = sorted(layer_by_depth.keys())
    n_depths = len(depths)
    
    if n_depths >= 3:
        sample_depths = [depths[0], depths[n_depths//2], depths[-1]]
    else:
        sample_depths = depths
    
    # Pick one attention and one MLP layer from each depth
    sample_layers = []
    for d in sample_depths:
        layers_at_depth = layer_by_depth[d]
        # Get attention layer (qkv or out)
        attn_layers = [l for l in layers_at_depth if "attn" in l or "qkv" in l or "w_q" in l]
        if attn_layers:
            sample_layers.append(attn_layers[0])
        # Get MLP layer (fc1, fc2, up, down)
        mlp_layers = [l for l in layers_at_depth if "mlp" in l or "fc" in l or "up" in l or "down" in l]
        if mlp_layers:
            sample_layers.append(mlp_layers[0])
    
    # Also add embedding/output layers if they exist
    for name in all_layer_names:
        if "embed" in name.lower() or "lm_head" in name.lower() or "output" in name.lower():
            if name not in sample_layers:
                sample_layers.append(name)
    
    print(f"\nSample layers selected ({len(sample_layers)} layers):")
    for l in sample_layers:
        print(f"  - {l}")
    
    # =========================================================================
    # ANALYZE SAMPLE LAYERS
    # =========================================================================
    
    print("\n" + "-"*70)
    print("INDIVIDUAL LAYER ANALYSIS")
    print("-"*70)
    
    sample_data = {}
    for layer_name in sample_layers:
        w = weights[layer_name]
        w_np = w.numpy()
        sv = compute_singular_values(w)
        
        # Compute stats
        energy = np.cumsum(sv**2) / np.sum(sv**2)
        eff_rank_99 = np.searchsorted(energy, 0.99) + 1
        stable_rank = np.sum(sv**2) / (sv[0]**2)
        condition = sv[0] / max(sv[-1], 1e-10)
        
        # Power law fit
        try:
            x_data = np.arange(1, len(sv) + 1)
            popt, _ = curve_fit(power_law, x_data[:len(sv)//2], sv[:len(sv)//2], 
                               p0=[sv[0], 0.5], maxfev=5000)
            power_exp = popt[1]
        except:
            power_exp = np.nan
        
        sample_data[layer_name] = {
            "weight": w_np,
            "sv": sv,
            "energy": energy,
            "shape": w_np.shape,
            "sigma_max": sv[0],
            "sigma_min": sv[-1],
            "condition": condition,
            "eff_rank_99": eff_rank_99,
            "eff_rank_ratio": eff_rank_99 / len(sv),
            "stable_rank": stable_rank,
            "power_exp": power_exp,
            "mean": np.mean(w_np),
            "std": np.std(w_np),
            "kurtosis": stats.kurtosis(w_np.flatten()),
            "skewness": stats.skew(w_np.flatten()),
        }
        
        print(f"\n{layer_name}:")
        print(f"  Shape: {w_np.shape}")
        print(f"  σ_max: {sv[0]:.4f}, σ_min: {sv[-1]:.6f}")
        print(f"  Condition number: {condition:.1f}")
        print(f"  Effective rank (99%): {eff_rank_99} / {len(sv)} ({eff_rank_99/len(sv)*100:.1f}%)")
        print(f"  Stable rank: {stable_rank:.1f}")
        print(f"  Power-law exponent: {power_exp:.3f}")
        print(f"  Weight std: {np.std(w_np):.6f}, kurtosis: {stats.kurtosis(w_np.flatten()):.2f}")
    
    # =========================================================================
    # PLOTS FOR SAMPLE LAYERS
    # =========================================================================
    
    n_samples = len(sample_layers)
    
    # Plot 1: Singular value decay for each sample layer
    fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(5 * ((n_samples + 1) // 2), 8))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    for i, layer_name in enumerate(sample_layers):
        data = sample_data[layer_name]
        sv = data["sv"]
        
        # Normalized SV decay
        axes[i].semilogy(sv / sv[0], color=colors[i], linewidth=2)
        
        # Fit power law line
        if not np.isnan(data["power_exp"]):
            x_fit = np.arange(1, len(sv) + 1)
            y_fit = power_law(x_fit, 1.0, data["power_exp"])
            axes[i].semilogy(x_fit - 1, y_fit, 'r--', alpha=0.7, 
                           label=f'power law (exp={data["power_exp"]:.2f})')
        
        # Mark 99% energy point
        axes[i].axvline(data["eff_rank_99"], color='green', linestyle=':', alpha=0.7,
                       label=f'99% energy @ {data["eff_rank_99"]}')
        
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        depth_str = ""
        for p in layer_name.split("."):
            if p.isdigit():
                depth_str = f"[L{p}] "
                break
        axes[i].set_title(f"{depth_str}{short_name}\n{data['shape']}", fontsize=10)
        axes[i].set_xlabel("SV Index")
        axes[i].set_ylabel("σ_i / σ_0")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sv_decay_samples.png"), dpi=150)
    plt.close()
    
    # Plot 2: Weight distributions for each sample layer
    fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(5 * ((n_samples + 1) // 2), 8))
    axes = axes.flatten()
    
    for i, layer_name in enumerate(sample_layers):
        data = sample_data[layer_name]
        w_flat = data["weight"].flatten()
        
        axes[i].hist(w_flat, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Gaussian fit
        x_range = np.linspace(w_flat.min(), w_flat.max(), 100)
        gaussian = stats.norm.pdf(x_range, data["mean"], data["std"])
        axes[i].plot(x_range, gaussian, 'r-', linewidth=2, label='Gaussian')
        
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        depth_str = ""
        for p in layer_name.split("."):
            if p.isdigit():
                depth_str = f"[L{p}] "
                break
        axes[i].set_title(f"{depth_str}{short_name}\nkurt={data['kurtosis']:.2f}, std={data['std']:.4f}", fontsize=10)
        axes[i].set_xlabel("Weight Value")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_dist_samples.png"), dpi=150)
    plt.close()
    
    # Plot 3: Cumulative energy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, layer_name in enumerate(sample_layers):
        data = sample_data[layer_name]
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        depth_str = ""
        for p in layer_name.split("."):
            if p.isdigit():
                depth_str = f"L{p}:"
                break
        
        # Normalize x-axis to [0, 1] for fair comparison
        x_norm = np.arange(len(data["energy"])) / len(data["energy"])
        ax.plot(x_norm, data["energy"], color=colors[i], linewidth=2, 
               label=f'{depth_str}{short_name} (eff_rank={data["eff_rank_ratio"]*100:.0f}%)')
    
    ax.axhline(0.99, color='red', linestyle='--', alpha=0.5, label='99% threshold')
    ax.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel("Normalized Rank (i / min(m,n))")
    ax.set_ylabel("Cumulative Energy")
    ax.set_title("Energy Concentration Across Sample Layers")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "energy_curves_samples.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Summary comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    layer_labels = []
    for layer_name in sample_layers:
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        depth_str = ""
        for p in layer_name.split("."):
            if p.isdigit():
                depth_str = f"L{p}:"
                break
        layer_labels.append(f"{depth_str}{short_name[:8]}")
    
    x = np.arange(len(sample_layers))
    
    # Effective rank ratio
    eff_ranks = [sample_data[l]["eff_rank_ratio"] * 100 for l in sample_layers]
    axes[0, 0].bar(x, eff_ranks, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel("Effective Rank %")
    axes[0, 0].set_title("Effective Rank (99% energy)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Condition number (log scale)
    conditions = [np.log10(sample_data[l]["condition"]) for l in sample_layers]
    axes[0, 1].bar(x, conditions, color=colors, edgecolor='black')
    axes[0, 1].set_ylabel("log10(Condition Number)")
    axes[0, 1].set_title("Condition Number")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Weight std
    stds = [sample_data[l]["std"] for l in sample_layers]
    axes[1, 0].bar(x, stds, color=colors, edgecolor='black')
    axes[1, 0].set_ylabel("Weight Std")
    axes[1, 0].set_title("Weight Standard Deviation")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Kurtosis
    kurtoses = [sample_data[l]["kurtosis"] for l in sample_layers]
    axes[1, 1].bar(x, kurtoses, color=colors, edgecolor='black')
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel("Kurtosis")
    axes[1, 1].set_title("Kurtosis (0 = Gaussian)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_comparison.png"), dpi=150)
    plt.close()
    
    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "="*70)
    print("KEY FINDINGS / GENERALIZATIONS")
    print("="*70)
    
    avg_eff_rank = np.mean([sample_data[l]["eff_rank_ratio"] for l in sample_layers])
    avg_power = np.nanmean([sample_data[l]["power_exp"] for l in sample_layers])
    avg_kurtosis = np.mean([sample_data[l]["kurtosis"] for l in sample_layers])
    
    findings = []
    
    if avg_eff_rank < 0.5:
        findings.append(f"✓ STRONG LOW-RANK: Only {avg_eff_rank*100:.0f}% of dimensions needed for 99% energy")
    elif avg_eff_rank < 0.8:
        findings.append(f"~ MODERATE LOW-RANK: {avg_eff_rank*100:.0f}% of dimensions for 99% energy")
    else:
        findings.append(f"✗ FULL-RANK: Weights are approximately full-rank")
    
    if avg_power > 0.5:
        findings.append(f"✓ FAST SV DECAY: Power-law exponent {avg_power:.2f}")
    else:
        findings.append(f"~ SLOW SV DECAY: Power-law exponent {avg_power:.2f}")
    
    if avg_kurtosis > 1:
        findings.append(f"✓ HEAVY-TAILED: Kurtosis {avg_kurtosis:.1f}")
    elif avg_kurtosis < -0.5:
        findings.append(f"~ LIGHT-TAILED: Kurtosis {avg_kurtosis:.1f}")
    else:
        findings.append(f"~ NEAR-GAUSSIAN: Kurtosis {avg_kurtosis:.1f}")
    
    # Check for depth trends
    depth_eff_ranks = []
    for l in sample_layers:
        for p in l.split("."):
            if p.isdigit():
                depth_eff_ranks.append((int(p), sample_data[l]["eff_rank_ratio"]))
                break
    
    if len(depth_eff_ranks) > 2:
        depths_arr = np.array([d[0] for d in depth_eff_ranks])
        ranks_arr = np.array([d[1] for d in depth_eff_ranks])
        corr, _ = stats.pearsonr(depths_arr, ranks_arr)
        if abs(corr) > 0.5:
            trend = "increases" if corr > 0 else "decreases"
            findings.append(f"✓ DEPTH TREND: Effective rank {trend} with depth (r={corr:.2f})")
    
    for f in findings:
        print(f"  {f}")
    
    print(f"\n  Plots saved to: {save_dir}/")
    
    return sample_data


if __name__ == "__main__":
    print("Testing the model loading and evaluation")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Single model analysis
    model_path = "/fast/slaing/exp/llm/tr_check_not_fused/job_idx_37/"
    model_name = "tr_check_not_fused"
    
    print(f"\nLoading {model_name}...")
    model, cfg = load_model(model_path, device)
    print(f"Config: optim={getattr(cfg, 'optim', 'N/A')}, lr={getattr(cfg, 'lr', 'N/A')}")


    #check out the dimensions of the embedding layer 

    





    """ 
    # Extract linear layer weights
    print("\nExtracting linear layer weights...")
    weights = get_linear_layers(model)
    print(f"Found {len(weights)} linear layers")
    
    # Generate comprehensive analysis
    output_dir = "./weight_analysis"
    sv_results, dist_results = generate_weight_report(weights, model_name, save_dir=output_dir)
    
    print("\nDone!")
    """ 