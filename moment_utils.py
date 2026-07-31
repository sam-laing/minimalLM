"""   
Considering the exponential moving average statistics being tracked during 
training for optimizers like adamw and custom

Was interested in the distribution of layer gradients
"""

import torch
from torch import Tensor  
import torch.distributions as dist
import matplotlib.pyplot as plt

def get_moment_dict(model, optimizer)->dict:
  """
  Iterates through the optimizer's state dict and records the exp_avg and
  exp_avg_sq tensors for each layer. Saves the tensors in a dictionary at full precision.

  Args:
      model
      optimizer (torch.optim.Optimizer): The AdamW optimizer.
      micro_step (int): The current micro step (used for naming the saved file).

  """
  param_to_name = {id(param): name for name, param in model.named_parameters()}

  moment_dict = {}
  for param_group in optimizer.param_groups:
    for param in param_group['params']:
      if param in optimizer.state:
        state = optimizer.state[param]
        if ('exp_avg' in state) and ('exp_avg_sq' in state):
          moment_dict[param_to_name[id(param)]] = {
              'exp_avg': state['exp_avg'].clone(),
              'exp_avg_sq': state['exp_avg_sq'].clone()
          }
  
  return moment_dict


def get_muon_momentum_dict(model, optimizer) -> dict:
  """
  Iterates through the optimizer's state dict and records the Muon
  momentum_buffer for each 2D (matrix) parameter, keyed by param name.
  1D params (norm gains, biases) are skipped: conditioning/SVD isn't
  meaningful for them.
  """
  param_to_name = {id(param): name for name, param in model.named_parameters()}

  momentum_dict = {}
  for group in optimizer.param_groups:
    for param in group['params']:
      state = optimizer.state.get(param, {})
      if 'momentum_buffer' in state and param.dim() == 2:
        momentum_dict[param_to_name[id(param)]] = state['momentum_buffer'].detach().clone()

  return momentum_dict


def stable_rank(matrix: Tensor) -> float:
  """
  (sum_i sigma_i^2) / sigma_max^2, normalized by min(shape) to lie in (0, 1].
  Low values = energy concentrated in a few directions ("ill-conditioned").
  More robust than raw condition number (sigma_max/sigma_min), which is
  dominated by whatever the smallest singular value happens to be.
  """
  s = torch.linalg.svdvals(matrix.float())
  raw = (s ** 2).sum() / (s[0] ** 2)
  return (raw / min(matrix.shape)).item()


def rank_momentum_conditioning(model, optimizer):
  """
  Returns [(param_name, normalized_stable_rank), ...] sorted ascending,
  i.e. most ill-conditioned momentum buffer first.
  """
  momentum_dict = get_muon_momentum_dict(model, optimizer)
  scored = [(name, stable_rank(buf)) for name, buf in momentum_dict.items()]
  return sorted(scored, key=lambda x: x[1])


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