import torch
import glog as log

def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
    return q_min, q_max

# CDF Quantization
def get_quantiles(tensor: torch.Tensor, n_bits) -> torch.Tensor:
    """
        Let f:X->Y be a cdf function
        then, given certain bins of the Y, we want to find
        the x\inX that are closest.
    """
    q_min, q_max = _get_quant_range(n_bits, False)
    q_range = q_max-q_min + 1
    sorted_tensor = torch.sort(tensor).values
    cdf = torch.arange(1, sorted_tensor.size(1) + 1)/sorted_tensor.size(1)
    cdf_bins = torch.linspace(0, 1, steps=q_range)
    quantile_bins = torch.zeros((sorted_tensor.size(0), cdf_bins.size(0)), device=tensor.device)
    # For every bin in cdf_bins, find the sorted_tensor value with a cdf closes to it.
    for i, bin in enumerate(cdf_bins):
        diffs =  (cdf-bin).abs()
        id = torch.argmin(diffs)
        quantile_bins[:, i]=sorted_tensor[:, id]
    return quantile_bins

def cdf_quantization(
        w: torch.tensor, n_bits, block_size
    ):
    w_view = w.view(-1, block_size)
    _, q_max = _get_quant_range(n_bits, sym=False)
    qmap = get_quantiles(w_view, n_bits)

    # Compute differences - w shape: (batch_size, block_size)
    # qmap shape: (batch_size, n_levels)
    differences = (w_view.unsqueeze(2) - qmap.unsqueeze(1)).abs()

    # Find indices of minimum differences
    quantized = differences.argmin(dim=2)  # Shape: (batch_size, block_size)

    # Get dequantized values using the computed indices
    # Create index for proper batch-wise gathering
    batch_indices = torch.arange(w_view.size(0), device=w_view.device).unsqueeze(1).expand(-1, w_view.size(1))
    dequantized = qmap[batch_indices, quantized]
    return quantized, qmap, dequantized

def cdf_quantization_legacy(
        w: torch.tensor, n_bits, block_size
    ):
    w_view = w.view(-1, block_size)
    _, q_max = _get_quant_range(n_bits, sym=False)
    qmap = get_quantiles(w_view, n_bits)
    differences = torch.ones_like(w_view)*q_max
    quantized = torch.zeros_like(w_view)
    dequantized = torch.zeros_like(w_view)
    for i in range(qmap.size(1)):
        qi = qmap[:,i][:, None].expand(-1, w_view.size(1))
        diff = (qi-w_view).abs()
        mask = differences>diff
        differences[mask]=diff[mask]
        quantized[mask] = i
        dequantized[mask] = qi[mask]
    return quantized, qmap, dequantized

def cdf_dequantization(
        w_q: torch.tensor, qmap
    ):
    return torch.tensor([qmap[i] for i in w_q.to(torch.int)])

# Gaussian quantization
def quantile_gaussian(p: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Compute the quantile function (inverse CDF) of a Gaussian distribution."""
    return mean + std * (2)**0.5 * torch.erfinv(2 * p - 1)

def nbit_to_qmap(n_bits):
    q_range = 2**n_bits
    # log.info(f"q_range {q_range}")
    cdf_bins = torch.linspace(0, 1, steps=q_range)
    return torch.clamp(quantile_gaussian(cdf_bins, 0, 1),-torch.inf, torch.inf)

def normal_quantization(w: torch.tensor, n_bits, block_size):
    w_view = w.view(-1, block_size)
    w_max_signed = w_view.amax(dim=1).clamp(min=1e-5)
    w_min_signed = w_view.amin(dim=1)
    qmap = nbit_to_qmap(n_bits).to(w_view.device)
    mean = w_view.mean(dim=1)
    std = w_view.std(dim=1)
    qmap = qmap.unsqueeze(0).expand(w_view.size(0), -1)  # Expand to match batch size, for free
    qmap = qmap * std.unsqueeze(1) + mean.unsqueeze(1)
    qmap = torch.clamp(qmap, w_min_signed.unsqueeze(1), w_max_signed.unsqueeze(1))
    differences = (w_view.unsqueeze(2) - qmap.unsqueeze(1)).abs()
    quantized = differences.argmin(dim=2)  # Shape: (batch_size, block_size)
    batch_indices = torch.arange(w_view.size(0), device=w_view.device).unsqueeze(1).expand(-1, w_view.size(1))
    dequantized = qmap[batch_indices, quantized]
    return quantized, (mean, std), dequantized

# Less memory usage
def normal_dequantization(
        w_q: torch.tensor, mean, std, sym,
        scales = None, base = None, n_bits=8
    ):
    qmap = nbit_to_qmap(n_bits, sym)
    outlier_heuristic = mean+2*std
    w_min_signed, w_max_signed = -outlier_heuristic, outlier_heuristic# However many stds you want
    qmap = torch.clamp(qmap*std+mean, w_min_signed, w_max_signed)
    return torch.tensor([qmap[i] for i in w_q.to(torch.int)]), qmap
