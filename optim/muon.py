import torch
import math
from torch.optim import Optimizer

#Newton Schulz approx: from KJ
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    X /= (X.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X

# fast nuclear norm via newton schultz algorithm
@torch.compile
def nuclear_norm_via_newtonschulz5(G, steps=6, eps=1e-7):
    """
    Fast approximation of nuclear norm via trace(sqrt(GG^T)) using
    NewtonSchulz iterations.

    seems like at least 6 steps needed to get below 2% approximation error

    """
    #take smaller matrix to iterate
    if G.size(0) <= G.size(1):
        A = G @ G.T   # (m, m)
    else:
        A = G.T @ G   # (n, n)
    #normalize into convergence region, remultiply later
    normA = A.norm(p='fro')
    A = A / (normA + eps)

    # classic NS for sqrt setup
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    Y = A
    Z = I
    for _ in range(steps):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    # multiply by norm again 
    sqrtA = Y * torch.sqrt(normA + eps)
    # retrn trace of sqrt
    return torch.trace(sqrtA)


# if full orthog is desired: slowwww for large matrices
@torch.compile
def get_svd(G, eps=1e-7):
    X = G.float()
    X /= (X.norm() + eps)

    if len(X.shape) != 2:
        raise ValueError("SVD expects a matrix")

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U, S.unsqueeze(0), Vh


def orthogonalise(G):
    transposed = False
    if G.size(0) > G.size(1):
        G = G.T
        transposed = True

    U, _, Vh = get_svd(G)
    out = U @ Vh
    return out.T if transposed else out

# the actual optimizer
class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.9,
        nesterov=False,
        ns_steps=5,
        eps=1e-7,
        orthogonalize=False,
        weight_decay=0.0,
        adjust_lr=True,
        dual_decay=False,
        sep_qkv=True,              
        adam_betas=(0.95, 0.95),
        adam_eps=1e-8,
        last_linear_adam=True, 
        embedding_layer_optim="muon", # can also have adamw or bernstein suggestion \ell_1 -> RMS operator norm: 
    ):
        """ 
        Some additional parts:
        - sep_qkv: optional QKV separation for Muon updates.... sep seems to matter a lot
        - AdamW for non muon parts in one optimizer
        - adjust_lr: as in Moonlight paper 
        
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            orthogonalize=orthogonalize,
            eps=eps,
            weight_decay=weight_decay,
            adjust_lr=adjust_lr,
            dual_decay=dual_decay,
            sep_qkv=sep_qkv,         
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            last_linear_adam=last_linear_adam,
            embedding_layer_optim=embedding_layer_optim,
        )

        super().__init__(params, defaults)
        self._optimizer_types = {}
        self._identify_optimizer_types()

    def _identify_optimizer_types(self):
        all_params = []
        for group in self.param_groups:
            all_params.extend(group["params"])

        last_param_id = id(all_params[-1]) if all_params else None
        last_linear_adam = self.param_groups[0].get("last_linear_adam", True)

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                # Detect embedding parameters (nn.Embedding: 2D, usually shape (vocab_size, dim) or (num_embeddings, embedding_dim))
                # Heuristic: if parameter is 2D and its shape[0] is much larger than shape[1], likely embedding
                if p.ndim == 2 and (p.shape[0] > 1000 and p.shape[0] > 2 * p.shape[1]):
                    self._optimizer_types[id(p)] = "embed"
                elif p.ndim == 2 and id(p) == last_param_id:
                    # Last linear layer
                    if last_linear_adam:
                        self._optimizer_types[id(p)] = "adam"
                    else:
                        self._optimizer_types[id(p)] = "muon"
                elif p.ndim == 2:
                    self._optimizer_types[id(p)] = "muon"
                else:
                    self._optimizer_types[id(p)] = "adam"

    def _muon_update_matrix(self, g, group, state):
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        orthogonalize = group["orthogonalize"]
        eps = group["eps"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)

        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        g_eff = g.add(buf, alpha=momentum) if nesterov else buf

        if orthogonalize:
            return orthogonalise(g_eff)
        else:
            return zeropower_via_newtonschulz5(g_eff, steps=ns_steps, eps=eps)


    def _muon_step(self, p, g, group, state):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        adjust_lr = group["adjust_lr"]
        dual_decay = group["dual_decay"]
        eps = group["eps"]
        sep_qkv = group["sep_qkv"]

        # optional QKV separation
        if (
            sep_qkv
            and p.ndim == 2
            and g.size(0) == 3 * g.size(1)
        ):
            d = g.size(1)

            g_q, g_k, g_v = g.split(d, dim=0)

            state_q = state.setdefault("q", {})
            state_k = state.setdefault("k", {})
            state_v = state.setdefault("v", {})

            upd_q = self._muon_update_matrix(g_q, group, state_q)
            upd_k = self._muon_update_matrix(g_k, group, state_k)
            upd_v = self._muon_update_matrix(g_v, group, state_v)

            update = torch.cat([upd_q, upd_k, upd_v], dim=0)

            effective_lr = lr
            if adjust_lr:
                effective_lr = 0.2 * math.sqrt(d) * lr
            if dual_decay:
                nuc_norm_approx = nuclear_norm_via_newtonschulz5(g.reshape(len(g), -1), steps=6, eps=eps)
                # normalize nuc norm by matrix size
                nuc_norm_approx = nuc_norm_approx / (min(g.shape[-2], g.shape[-1]))
                effective_lr *= nuc_norm_approx.clamp(min=0.1)

            if weight_decay > 0:
                p.data.add_(p.data, alpha=-weight_decay * lr)

            p.data.add_(update, alpha=-effective_lr)
            return

        update = self._muon_update_matrix(
            g.reshape(len(g), -1), group, state
        ).view(g.shape)

        effective_lr = lr
        if adjust_lr:
            effective_lr = 0.2 * math.sqrt(max(g.size(-2), g.size(-1))) * lr

        if dual_decay:
            nuc_norm_approx = nuclear_norm_via_newtonschulz5(g.reshape(len(g), -1), steps=6, eps=eps)
            # normalize nuc norm by matrix size
            nuc_norm_approx = nuc_norm_approx / (min(g.shape[-2], g.shape[-1]))
            effective_lr *= nuc_norm_approx.clamp(min=0.1) 
        if weight_decay > 0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

        p.data.add_(update, alpha=-effective_lr)
    


    def _adam_step(self, p, g, group, state):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps = group["adam_eps"]

        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        p.data.addcdiv_(exp_avg, denom, value=-step_size)

        if weight_decay > 0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

    def _embedding_layer_step(self, p, g, group, state):
        lr = group["lr"]
        eps = group["eps"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["weight_decay"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)

        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        g_eff = g.add(buf, alpha=momentum) if nesterov else buf

        # steepest descent under the \ell1->RMS operator norm:
        # normalize each column by its RMS
        col_rms = torch.sqrt((g_eff ** 2).mean(dim=0, keepdim=True) + eps)  # (1, vocab)
        update = g_eff / col_rms  # (embed_dim, vocab) — each col has unit RMS

        # lr scaling: match Muon's adjust_lr spirit
        # sqrt(embed_dim) keeps the effective step size consistent with weight matrices
        if group["adjust_lr"]:
            effective_lr = 0.2 * math.sqrt(g.size(0)) * lr
        else:
            effective_lr = lr

        if weight_decay > 0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

        p.data.add_(update, alpha=-effective_lr)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                opt_type = self._optimizer_types.get(id(p), "muon")

                if opt_type == "embed":
                    self._embedding_layer_step(p, g, group, state)
                elif opt_type == "muon":
                    self._muon_step(p, g, group, state)
                else:
                    self._adam_step(p, g, group, state)
