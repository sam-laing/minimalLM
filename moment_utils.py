"""   
Considering the exponential moving average statistics being tracked during 
training for optimizers like adamw and custom

Was interested in the distribution of layer gradients
"""

import torch
from torch import Tensor  
import torch.distributions as dist
import matplotlib.pyplot as plt

def get_histogram_for_adam_layer(layer_tensor, n_bins=100):
    """Returns normalized histogram and bin edges"""
    min_elt, max_elt = layer_tensor.min(), layer_tensor.max()
    hist = torch.histc(layer_tensor, bins=n_bins, min=min_elt, max=max_elt)
    hist = hist / hist.sum()
    # Add one more bin edge to match histogram size
    bin_edges = torch.linspace(min_elt, max_elt, n_bins + 1)
    return hist, bin_edges

def get_mean_and_variance(layer_tensor: Tensor):
    # if shape is more than 1D then put into 1D
    if len(layer_tensor.shape) > 1:
        layer_tensor = layer_tensor.view(-1)
    """Returns mean and variance of tensor"""
    return layer_tensor.mean(), layer_tensor.var(unbiased=True)

def compute_kl_div_from_normal(layer_tensor: Tensor, nbins: int = 100, plot: bool = False):
    """Compute KL divergence between tensor distribution and matching normal"""
    hist, bin_edges = get_histogram_for_adam_layer(layer_tensor, n_bins=nbins)
    
    # Compute bin centers - now matches hist size
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get mean and variance
    mean, variance = get_mean_and_variance(layer_tensor)
    
    # Create normal distribution and compute PDF
    normal_dist = dist.Normal(mean, torch.sqrt(variance))
    pdf = torch.exp(normal_dist.log_prob(bin_centers))
    pdf = pdf / pdf.sum()
    
    # Verify shapes match
    assert hist.shape == pdf.shape, f"Shape mismatch: hist {hist.shape}, pdf {pdf.shape}"
    
    # Compute KL Divergence
    mask = (hist > 0) & (pdf > 0)
    kl_div = torch.sum(hist[mask] * (torch.log(hist[mask]) - torch.log(pdf[mask])))
    
    return kl_div.item()

def compute_kl_div_from_chi_squared(layer_tensor: Tensor, df: float = 2.0, nbins: int = 100, plot: bool = False):
    """Compute KL divergence between tensor distribution and chi-squared"""
    hist, bin_edges = get_histogram_for_adam_layer(layer_tensor, n_bins=nbins)
    
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create chi-squared distribution
    # Note: chi-squared is always non-negative, so we shift the data if needed
    shifted_centers = bin_centers - bin_centers.min() if bin_centers.min() < 0 else bin_centers
    chi_squared_dist = dist.Chi2(df=torch.tensor([df]))
    
    # Compute PDF for chi-squared
    pdf = torch.exp(chi_squared_dist.log_prob(shifted_centers))
    pdf = pdf / pdf.sum()  # Normalize
    
    # Verify shapes match
    assert hist.shape == pdf.shape, f"Shape mismatch: hist {hist.shape}, pdf {pdf.shape}"
    
    # Compute KL Divergence
    mask = (hist > 0) & (pdf > 0)
    kl_div = torch.sum(hist[mask] * (torch.log(hist[mask]) - torch.log(pdf[mask])))
    
    return kl_div.item()

def create_plot(hist, hist_bins, title, path = None, show=None):
    """Create and save histogram plot"""
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plt.bar(bin_centers, hist, width=(hist_bins[1] - hist_bins[0]), edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    if show is not None:
        plt.show()
    plt.close()