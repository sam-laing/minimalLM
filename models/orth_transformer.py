"""Transformer++, a simple LLama-style Transformer, supporting RMSNorm, RoPE, GLU

Includes orthogonal initialization strategies for transformer weight matrices.

Three modes:
  1. "orthogonal"       — pure scaled orthogonal Q
  2. "perturbed"        — W* + radius * Q  (orthogonal perturbation around a target)
  3. "kaiming_orthog"   — Kaiming-scaled orthogonal (preserves variance like kaiming but with orthogonal structure)

The key insight for Muon: since Muon projects gradients onto the Stiefel manifold
(orthonormal matrices) via Newton-Schulz, starting near that manifold means the
optimizer doesn't waste early steps reshaping the singular value spectrum.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

from .components import RMSNorm, MLP, GLU, MLPReluSquared
from .embeddings import precompute_freqs_cis, apply_rotary_emb_complex_like


class InitMode(str, Enum):
    ORTHOGONAL = "orthogonal"
    PERTURBED = "perturbed"
    KAIMING_ORTHOG = "kaiming_orthog"


def _orthogonal_matrix(rows: int, cols: int, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """Generate an orthogonal (or semi-orthogonal) matrix via QR decomposition.
    
    For non-square matrices, returns Q from the QR decomposition of a
    random Gaussian matrix, giving a uniformly-distributed element of
    the Stiefel manifold V_{min(m,n)}(R^{max(m,n)}).
    """
    mat = torch.randn(rows, cols, device=device, dtype=dtype)
    if rows >= cols:
        Q, R = torch.linalg.qr(mat)
        # Make QR unique by ensuring diag(R) > 0
        sign = torch.sign(torch.diag(R))
        sign[sign == 0] = 1
        Q = Q * sign.unsqueeze(0)
    else:
        Q, R = torch.linalg.qr(mat.T)
        sign = torch.sign(torch.diag(R))
        sign[sign == 0] = 1
        Q = (Q * sign.unsqueeze(0)).T
    return Q


def orthogonal_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Pure orthogonal init: W = gain * Q.
    
    For 2D weight matrices this is straightforward. For >2D (e.g., the fused
    w_qkv which is (3*dim, dim)), we reshape to 2D, orthogonalize, reshape back.
    
    All singular values will be exactly `gain`.
    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError("Orthogonal init requires ndim >= 2")
        
        rows, cols = tensor.shape[0], tensor[0].numel()  # handles >2D
        Q = _orthogonal_matrix(rows, cols, device=tensor.device, dtype=tensor.dtype)
        tensor.copy_((gain * Q).view_as(tensor))
    return tensor


def kaiming_orthogonal_init_(tensor: torch.Tensor, nonlinearity: str = 'relu') -> torch.Tensor:
    """Kaiming-scaled orthogonal init: W = sqrt(2 / fan_in) * Q.
    
    This gives you the variance-preserving property of Kaiming init
    (correct forward-pass signal magnitude for ReLU/SiLU networks)
    but with perfectly conditioned singular values (condition number = 1).
    
    Standard Kaiming draws iid Gaussian entries with std = sqrt(2/fan_in),
    which gives singular values following a Marchenko-Pastur distribution.
    Here we replace the random matrix with an orthogonal one scaled to
    match the same operator norm / Frobenius norm.
    
    For SiLU (used in your MLP/GLU), the recommended gain is ~sqrt(2)
    same as ReLU in practice.
    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError("Kaiming orthogonal init requires ndim >= 2")
        
        fan_in = tensor[0].numel()  # product of all dims except dim 0
        
        # Gain lookup (matching torch.nn.init.calculate_gain)
        gains = {
            'linear': 1.0,
            'relu': math.sqrt(2.0),
            'silu': math.sqrt(2.0),  # empirically similar to relu
            'tanh': 5.0 / 3.0,
            'sigmoid': 1.0,
        }
        gain = gains.get(nonlinearity, 1.0)
        
        # Kaiming std = gain / sqrt(fan_in)
        # For orthogonal matrix Q with singular values all = 1,
        # we want ||Wx||^2 ≈ (gain^2 / fan_in) * ||x||^2
        # So scale = gain / sqrt(fan_in)
        scale = gain / math.sqrt(fan_in)
        
        rows, cols = tensor.shape[0], fan_in
        Q = _orthogonal_matrix(rows, cols, device=tensor.device, dtype=tensor.dtype)
        tensor.copy_((scale * Q).view_as(tensor))
    return tensor


def perturbed_orthogonal_init_(
    tensor: torch.Tensor,
    W_star: Optional[torch.Tensor] = None,
    radius: float = 0.01,
    gain: float = 1.0,
    w_star_mode: Literal['zeros', 'kaiming', 'custom'] = 'kaiming'
) -> torch.Tensor:
    """Perturbed init: W = W* + radius * Q for orthogonal Q.
    
    This lets you start near a known-good solution W* while adding a
    controlled orthogonal perturbation. Ideas for W*:
    
      - 'zeros':   W* = 0, so W = radius * Q. Useful for output projections
                   where you want near-zero init (like your residual scaling).
      - 'kaiming': W* = kaiming_uniform init. Adds orthogonal structure on top
                   of the standard init.
      - 'custom':  Pass your own W_star tensor (e.g., from a pretrained checkpoint,
                   or a structured matrix like a DFT/Hadamard basis).
    
    The radius controls how far you wander from W*. For Muon, think of this
    as: W* sets the energy landscape neighborhood, and the orthogonal Q
    ensures you explore it with maximally diverse directions.
    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError("Perturbed orthogonal init requires ndim >= 2")
        
        rows, cols = tensor.shape[0], tensor[0].numel()
        
        # Build W*
        if w_star_mode == 'custom':
            if W_star is None:
                raise ValueError("Must provide W_star tensor when w_star_mode='custom'")
            assert W_star.shape == tensor.shape, f"W_star shape {W_star.shape} != tensor shape {tensor.shape}"
            w_star_flat = W_star.view(rows, cols)
        elif w_star_mode == 'zeros':
            w_star_flat = torch.zeros(rows, cols, device=tensor.device, dtype=tensor.dtype)
        elif w_star_mode == 'kaiming':
            w_star_flat = tensor.view(rows, cols).clone()  # use whatever init is already there
            fan_in = cols
            std = math.sqrt(2.0) / math.sqrt(fan_in)
            w_star_flat.normal_(0, std)
        else:
            raise ValueError(f"Unknown w_star_mode: {w_star_mode}")
        
        Q = _orthogonal_matrix(rows, cols, device=tensor.device, dtype=tensor.dtype)
        result = w_star_flat + radius * gain * Q
        tensor.copy_(result.view_as(tensor))
    return tensor


# ─────────────────────────────────────────────────────────
# Integration helpers for Transformer
# ─────────────────────────────────────────────────────────

def init_transformer_weights(
    model: nn.Module,
    mode: str = "kaiming_orthog",
    n_layers: int = 1,
    nonlinearity: str = "silu",
    gain: float = 0.02,
    perturb_radius: float = 0.01,
    perturb_w_star_mode: str = "kaiming",
):
    """Apply orthogonal init to all Linear layers in a Transformer.
    
    Handles the residual branch scaling automatically:
    output projections (w_out, fc2) get an additional 1/sqrt(2*n_layers) factor.
    
    Args:
        model: The transformer module
        mode: One of "orthogonal", "kaiming_orthog", "perturbed"
        n_layers: Number of transformer blocks (for residual scaling)
        nonlinearity: Activation function name for kaiming gain calculation
        perturb_radius: Radius for perturbed mode
        perturb_w_star_mode: W* strategy for perturbed mode
    """
    residual_scale = 1.0 / math.sqrt(2 * n_layers)
    
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue  # skip norm weights, biases
            
        # Determine if this is a residual output projection
        is_residual_output = name.endswith('w_out.weight') or name.endswith('fc2.weight')
        
        if mode == "orthogonal":
            g = gain
            if is_residual_output:
                g *= residual_scale
            orthogonal_init_(param, gain=g)
            
        elif mode == "kaiming_orthog":
            kaiming_orthogonal_init_(param, nonlinearity=nonlinearity)
            if is_residual_output:
                param.data *= residual_scale
                
        elif mode == "perturbed":
            gain = 1.0
            if is_residual_output:
                gain = residual_scale
            perturbed_orthogonal_init_(
                param,
                radius=perturb_radius,
                gain=gain,
                w_star_mode=perturb_w_star_mode,
            )
        else:
            raise ValueError(f"Unknown init mode: {mode}")
    
    # Embeddings: always normal init (orthogonal doesn't make sense for embeddings)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    n_heads: int
    mlp: str = 'mlp'
    rmsorm_eps: float = 1e-6
    tie_embeddings: bool = False
    # Init options: "normal" (standard), "orthogonal", "kaiming_orthog", "perturbed"
    init_mode: str = 'kaiming_orthog'
    init_gain: float = 0.02            # gain for "orthogonal" mode
    init_nonlinearity: str = 'silu'    # nonlinearity for kaiming gain calc
    perturb_radius: float = 0.01       # radius for "perturbed" mode
    perturb_w_star_mode: str = 'kaiming'  # W* strategy: 'zeros', 'kaiming', 'custom'


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}


class OrthogonalLinearInit:
    def __init__(self, gain=0.02):
        self.gain = gain
    
    def __call__(self, tensor: torch.Tensor):
        assert tensor.ndim >= 2, "Only tensors with 2 or more dimensions are supported"
        
        if tensor.ndim > 2:
            flat_shape = (tensor.shape[0], -1)
            flat_tensor = tensor.view(flat_shape)
            nn.init.orthogonal_(flat_tensor, gain=self.gain)

        #return to normal shape ... like if view is done go back to actual dimension  
        tensor.copy_(flat_tensor.view(tensor.shape))
        return tensor

    def orthogonal_init(self, tensor: torch.Tensor):
        with torch.no_grad():
            if tensor.ndim < 2:
                raise ValueError("Only tensors with 2 or more dimensions are supported")
            elif tensor.ndim == 2:
                nn.init.orthogonal_(tensor, gain=self.gain)
            else:
                # For tensors with more than 2 dimensions, we can initialize each slice orthogonally
                flat_shape = (tensor.shape[0], -1)
                flat_tensor = tensor.view(flat_shape)
                nn.init.orthogonal_(flat_tensor, gain=self.gain)
                tensor.copy_(flat_tensor.view(tensor.shape))

    def soft_orthogonal_init(self, tensor: torch.Tensor, W_star: torch.Tensor, rad:float=0.001):
        #combine this orthogonal init but have something like W^* + rad Q for orthogonal Q and W^*
        pass

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        
        self.w_qkv = nn.Linear(cfg.dim, 3*cfg.dim, bias=False)
        self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
    
    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape # (bsz, seqlen, d)
        
        q, k, v = self.w_qkv(x).split(d, dim=2) # (bsz, seqlen, d)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis) # (bsz, seqlen, nh, h_dim)
        
        q = q.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        k = k.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        v = v.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (bsz, nh, seqlen, h_dim)
        
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d) # (bsz, seqlen, d)
        
        return self.w_out(out)

class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.layer_id = layer_id
    
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg.n_layers
        head_dim = cfg.dim // cfg.n_heads; assert cfg.dim % cfg.n_heads == 0
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(idx, cfg) for idx in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0:cfg.seq_len]
        
        # Weight initialisation (configurable via cfg.init_mode)
        if cfg.init_mode == 'normal':
            # Standard GPT-style init (same as transformer.py)
            self.apply(self._init_weights)
            self._scale_residual_branches()
        else:
            init_transformer_weights(
                self,
                mode=cfg.init_mode,
                n_layers=cfg.n_layers,
                nonlinearity=cfg.init_nonlinearity,
                gain=cfg.init_gain,
                perturb_radius=cfg.perturb_radius,
                perturb_w_star_mode=cfg.perturb_w_star_mode,
            )
        
        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x):
        # x: (bsz, seqlen)
        x = self.embed_tokens(x) # (bsz, seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(x.device)
        for layer in self.layers:
            x = layer(x, self.freqs_cis) # (bsz, seqlen, dim)
        return self.lm_head(self.out_norm(x)) # (bsz, seqlen, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_branches(self):
        for n, p in self.named_parameters():
            if n.endswith('fc2.weight'): # mlp/glu output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
            if n.endswith('w_out.weight'): # attn output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if not self.lm_head.weight is self.embed_tokens.weight:  # if no weight tying
                n_params -= self.lm_head.weight.numel()
        return n_params


