
import os
import random
import numpy as np
import torch

from torch.distributed import init_process_group, destroy_process_group

from utils import print_master

from torch import Tensor
import matplotlib.pyplot as plt
import torch.distributions as dist


def pytorch_setup(cfg):
  """Returns device, rank, seed, etc and initialize DDP"""
  ddp = int(os.environ.get('RANK', -1)) != -1  # check if DDP is enabled

  if ddp:
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.device(device)
    master_process = (rank == 0)
    seed_offset = rank
  else:
    master_process = True
    seed_offset = 0
    local_rank = None
    world_size = 1
    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda'
    elif torch.backends.mps.is_available():
      device = 'mps'  # NOTE: macOS metal support to be tested

  random.seed(cfg.seed + seed_offset)
  np.random.seed(cfg.seed + seed_offset)
  torch.manual_seed(cfg.seed + seed_offset)
  
  # allow TF32
  torch.backends.cuda.matmul.allow_tf32 = getattr(cfg, 'cuda_allow_tf32', False)
  torch.backends.cudnn.allow_tf32 = getattr(cfg, 'cudnn_allow_tf32', False)

  # limit CUDA memory
  if hasattr(cfg, 'set_memory_fraction'):
    tot_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    red_mem_gb = tot_mem_gb * cfg.set_memory_fraction
    print_master(f"Limit GPU memory from {tot_mem_gb:.2f}GB to: {red_mem_gb:.2f}GB")
    torch.cuda.set_per_process_memory_fraction(cfg.set_memory_fraction, device=device)

  # deterministic run
  if getattr(cfg, 'determinisitc', False):
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark = False
    
  return local_rank, world_size, device, master_process


def destroy_ddp():
  if torch.distributed.is_initialized():
    torch.distributed.barrier()
    destroy_process_group()



# lookings at some distribution stuff
def get_histogram_for_adam_layer(layer_tensor, n_bins=100):
  """
  Returns the histogram of the layer's gradient tensor.
  normalises the histogram to make it a probability distribution.
  
  """
  min_elt, max_elt = layer_tensor.min(), layer_tensor.max()
  hist = torch.histc(layer_tensor, bins=n_bins,min=min_elt, max=max_elt)
  hist = hist / hist.sum()
  bin_edges = torch.linspace(min_elt, max_elt, n_bins)
  return hist, bin_edges


def get_moments_dict(model, optimizer) -> dict:
    """  
    Returns a dictionary of the first and second moments of the optimizer's 
    moving averages for each layer
    """
    # Handle DataParallel or DistributedDataParallel
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module

    param_to_name = {id(param): name for name, param in model.named_parameters()}

    # Dictionary to store exp_avg and exp_avg_sq
    moments_dict = {}

    # Iterate through the parameter groups in the optimizer
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                layer_name = param_to_name.get(id(p), "Unknown Layer")
                # Remove 'module.' prefix if it exists
                if layer_name.startswith('module.'):
                    layer_name = layer_name[len('module.'):]
                moments_dict[layer_name] = {
                    'exp_avg': state['exp_avg'],
                    'exp_avg_sq': state['exp_avg_sq']
                }
            else:
                print(f"State for parameter {param_to_name.get(id(p), 'Unknown Layer')} does not contain 'exp_avg' or 'exp_avg_sq'")

    return moments_dict




def save_layer_histogram_plots(epoch, moments_dict, layer_name, savepath):
    """
    Given the dictionary of moments for each layer, plot a histogram of the 
    exp_avg and exp_avg_sq for the specified layer and save plots in the correct folder
    """
    try:
        layer_moments = moments_dict.get(layer_name, None)
        if layer_moments is None:
            print(f"Layer {layer_name} not found in moments_dict")
            return
        
        # Ensure the directories exist
        exp_avg_path = os.path.join(savepath, 'exp_avg')
        exp_avg_sq_path = os.path.join(savepath, 'exp_avg_sq')
        os.makedirs(exp_avg_path, exist_ok=True)
        os.makedirs(exp_avg_sq_path, exist_ok=True)

        # Get the exp_avg and exp_avg_sq tensors
        exp_avg = layer_moments['exp_avg']
        exp_avg_sq = layer_moments['exp_avg_sq']

        # Get the histogram for the exp_avg and exp_avg_sq tensors
        exp_avg_hist, exp_avg_bin_edges = get_histogram_for_adam_layer(exp_avg)
        exp_avg_sq_hist, exp_avg_sq_bin_edges = get_histogram_for_adam_layer(exp_avg_sq)

        # Plot the histograms
        plt.figure()
        plt.bar(exp_avg_bin_edges[:-1], exp_avg_hist, width=exp_avg_bin_edges[1] - exp_avg_bin_edges[0], alpha=0.5) 
        plt.title(f'Exp Avg Histogram for {layer_name}')
        plt.xlabel('Exp Avg')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(exp_avg_path, f'{layer_name}_exp_avg_epoch_{epoch}.png'))
        plt.close()

        plt.figure()
        plt.bar(exp_avg_sq_bin_edges[:-1], exp_avg_sq_hist, width=exp_avg_sq_bin_edges[1] - exp_avg_sq_bin_edges[0], alpha=0.5)
        plt.title(f'Exp Avg Sq Histogram for {layer_name}')
        plt.xlabel('Exp Avg Sq')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(exp_avg_sq_path, f'{layer_name}_exp_avg_sq_epoch_{epoch}.png'))
        plt.close()

    except Exception as e:
        print(f"Error in save_layer_histogram_plots: {e}")


def save_all_layer_histogram_plots(epoch, moments_dict, savepath):
    """
    Given the dictionary of moments for each layer, plot a histogram of the 
    exp_avg and exp_avg_sq for each layer and save plots in the correct folder
    """
    try:
        for layer_name, layer_moments in moments_dict.items():
            if layer_moments is None:
                print(f"Layer {layer_name} not found in moments_dict")
                continue
            
            # Ensure the directories exist
            exp_avg_path = os.path.join(savepath, 'exp_avg')
            exp_avg_sq_path = os.path.join(savepath, 'exp_avg_sq')
            os.makedirs(exp_avg_path, exist_ok=True)
            os.makedirs(exp_avg_sq_path, exist_ok=True)

            # Get the exp_avg and exp_avg_sq tensors
            exp_avg = layer_moments['exp_avg']
            exp_avg_sq = layer_moments['exp_avg_sq']

            # Get the histogram for the exp_avg and exp_avg_sq tensors
            exp_avg_hist, exp_avg_bin_edges = get_histogram_for_adam_layer(exp_avg)
            exp_avg_sq_hist, exp_avg_sq_bin_edges = get_histogram_for_adam_layer(exp_avg_sq)

            # Plot the histograms
            plt.figure()
            plt.bar(exp_avg_bin_edges[:-1], exp_avg_hist, width=exp_avg_bin_edges[1] - exp_avg_bin_edges[0], alpha=0.5) 
            plt.title(f'Exp Avg Histogram for {layer_name}')
            plt.xlabel('Exp Avg')
            plt.ylabel('Probability')
            plt.savefig(os.path.join(exp_avg_path, f'{layer_name}_exp_avg_epoch_{epoch}.png'))
            plt.close()

            plt.figure()
            plt.bar(exp_avg_sq_bin_edges[:-1], exp_avg_sq_hist, width=exp_avg_sq_bin_edges[1] - exp_avg_sq_bin_edges[0], alpha=0.5)
            plt.title(f'Exp Avg Sq Histogram for {layer_name}')
            plt.xlabel('Exp Avg Sq')
            plt.ylabel('Probability')
            plt.savefig(os.path.join(exp_avg_sq_path, f'{layer_name}_exp_avg_sq_epoch_{epoch}.png'))
            plt.close()

    except Exception as e:
        print(f"Error in save_all_layer_histogram_plots: {e}")

def get_mean_and_variance(layer_tensor:Tensor):
  """
  Returns the mean and variance of a layer tensor
  """
  return layer_tensor.mean().item(), layer_tensor.var(unbiased=True).item()

def compute_kl_div_from_normal(layer_tensor:Tensor, nbins:int = 100, plot: bool = False):
  """
  create a pdf from the layer tensor and compare it to a
  normal distribution with the same mean and variance
  
  """

  N = layer_tensor.shape[0]
  hist, bin_edges = get_histogram_for_adam_layer(layer_tensor, n_bins=nbins)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

  # get mean and variance
  mean, variance= get_mean_and_variance(layer_tensor)

  # create a normal distribution
  normal_dist = dist.Normal(mean, torch.sqrt(variance))
  pdf = torch.exp(normal_dist.log_prob(bin_centers))
  pdf = pdf / pdf.sum()


  # Compute KL Divergence
  mask = (hist>0) & (pdf>0)
  kl_div = torch.sum(hist[mask] * torch.log(hist[mask]) - torch.log(pdf[mask]))

  return kl_div.item()




import scipy.stats as stats
def ks_test(layer_tensor:Tensor):
    """   
    Perform the Kolmogorov-Smirnov test to see if the data is normally distributed.
    """
    mu, sig = layer_tensor.mean().item(), layer_tensor.std().item()
    x_cpu = layer_tensor.cpu().numpy()
    return stats.kstest(x_cpu, 'norm', args=(mu, sig))


def compute_kl_div_and_ks_test_for_all_layers(moments_dict):
    """
    Compute the KL divergence and perform the Kolmogorov-Smirnov test for exp_avg of each layer.
    Return a JSON-like object with all the metrics.
    """
    metrics = {}
    try:
        for layer_name, layer_moments in moments_dict.items():
            if layer_moments is None:
                print(f"Layer {layer_name} not found in moments_dict")
                continue

            # Get the exp_avg tensor
            exp_avg = layer_moments['exp_avg']

            # Compute KL divergence
            kl_div_exp_avg = compute_kl_div_from_normal(exp_avg)

            # Perform KS test
            ks_test_exp_avg = ks_test(exp_avg)

            # Get mean and variance
            exp_avg_mean, exp_avg_var = get_mean_and_variance(exp_avg)

            # Store metrics in the dictionary
            metrics[layer_name] = {
                "kl_divergence_exp_avg": kl_div_exp_avg,
                "ks_test_statistic_exp_avg": ks_test_exp_avg.statistic,
                "ks_test_pvalue_exp_avg": ks_test_exp_avg.pvalue,
                "exp_avg_mean": exp_avg_mean,
                "exp_avg_var": exp_avg_var
            }

    except Exception as e:
        print(f"Error in compute_kl_div_and_ks_test_for_all_layers: {e}")

    return metrics


