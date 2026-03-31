import torch
import math
from torch.optim import Optimizer

# Newton Schulz approx: from KJ
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


# Polar Express: matrix sign / polar factor via tuned degree-5 polynomials
# https://arxiv.org/abs/2505.16932
ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563,   -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106,  0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,   0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,   0.51004894012372),
    (2.300652019954817,   -1.6689039845747493,  0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868,  0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487,  0.3750001645474248),
    (1.875,               -1.25,                0.375),
]
# safety factor for numerical stability (exclude last polynomial)
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


@torch.compile
@torch.no_grad()
def zeropower_via_polar_express(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Polar Express algorithm for the matrix sign / polar factor.
    https://arxiv.org/abs/2505.16932

    Drop-in replacement for zeropower_via_newtonschulz5: same input/output
    contract (2D matrix in, orthogonal factor out), controlled via steps.
    The 8 pre-computed ABC entries cover most of the convergence; beyond that
    the last entry (converged regime) repeats automatically.
    """
    assert G.ndim == 2
    should_transpose: bool = G.size(0) > G.size(1)
    x = G.bfloat16()
    if should_transpose:
        x = x.mT
    x /= x.norm() * 1.01
    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        # compute x = (aI + (bI + cS)S) x  — avoids materialising extra temporaries
        y = c * s
        y.diagonal().add_(b)
        y = y @ s
        y.diagonal().add_(a)
        x = y @ x
    if should_transpose:
        x = x.mT
    x = torch.nan_to_num(x)
    return x.float()


@torch.compile
def nuclear_norm_via_newtonschulz5(G, steps=6, eps=1e-7):
    """
    Fast approximation of nuclear norm via trace(sqrt(GG^T)) using
    NewtonSchulz iterations.

    seems like at least 6 steps needed to get below 2% approximation error
    """
    if G.size(0) <= G.size(1):
        A = G @ G.T   # (m, m)
    else:
        A = G.T @ G   # (n, n)

    normA = A.norm(p='fro')
    A = A / (normA + eps)

    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    Y = A
    Z = I
    for _ in range(steps):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    sqrtA = Y * torch.sqrt(normA + eps)
    return torch.trace(sqrtA)


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
        orthog_method="newtonschulz",  # "newtonschulz" | "polar_express"
        polar_steps=10,                # steps used when orthog_method="polar_express"
        weight_decay=0.0,
        adjust_lr=True,
        dual_decay=False,
        sep_qkv=True,
        adam_betas=(0.95, 0.95),
        adam_eps=1e-8,
        last_linear_adam=True,
        embedding_layer_optim="muon",  # "muon" | "adam" | "rms"
    ):
        """
        Some additional parts:
        - orthog_method: which algorithm to use for the orthogonalization step.
            "newtonschulz"  — degree-5 polynomial NS (fast, default)
            "polar_express" — Polar Express (https://arxiv.org/abs/2505.16932),
                              tuned degree-5 polynomial coefficients with 8
                              pre-computed ABC entries; controlled via polar_steps
                              (default 10; beyond 8 the last entry repeats)
        - polar_steps: number of iterations for polar_express (default 10)
        - sep_qkv: optional QKV separation for Muon updates
        - AdamW for non-muon parts in one optimizer
        - adjust_lr: as in Moonlight paper
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            orthogonalize=orthogonalize,
            orthog_method=orthog_method,
            polar_steps=polar_steps,
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
        # Find the last 2D weight parameter (lm_head) from decay_params (first group).
        # param_groups = [decay_params, no_decay_params], so all_params[-1] would be
        # a norm layer, not lm_head.
        last_2d_param_id = None
        for group in self.param_groups:
            if group.get("weight_decay", 0) > 0:
                for p in group["params"]:
                    if p.ndim == 2:
                        last_2d_param_id = id(p)

        last_linear_adam = self.param_groups[0].get("last_linear_adam", True)

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if p.ndim == 2 and id(p) == last_2d_param_id:
                    self._optimizer_types[id(p)] = "adam" if last_linear_adam else "muon"
                elif p.ndim == 2 and (p.shape[0] > 1000 and p.shape[0] > 2 * p.shape[1]):
                    self._optimizer_types[id(p)] = "embed"
                elif p.ndim == 2:
                    self._optimizer_types[id(p)] = "muon"
                else:
                    self._optimizer_types[id(p)] = "adam"

    def _muon_update_matrix(self, g, group, state):
        momentum      = group["momentum"]
        nesterov      = group["nesterov"]
        ns_steps      = group["ns_steps"]
        orthogonalize = group["orthogonalize"]
        orthog_method = group["orthog_method"]
        polar_steps   = group["polar_steps"]
        eps           = group["eps"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)

        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        g_eff = g.add(buf, alpha=momentum) if nesterov else buf
        g_eff = g_eff.to(g.device)

        if orthogonalize:
            return orthogonalise(g_eff)
        elif orthog_method == "polar_express":
            return zeropower_via_polar_express(g_eff, steps=polar_steps)
        else:  # "newtonschulz" (default)
            return zeropower_via_newtonschulz5(g_eff, steps=ns_steps, eps=eps)

    def _muon_step(self, p, g, group, state):
        lr           = group["lr"]
        weight_decay = group["weight_decay"]
        adjust_lr    = group["adjust_lr"]
        dual_decay   = group["dual_decay"]
        eps          = group["eps"]
        sep_qkv      = group["sep_qkv"]

        # optional QKV separation
        if sep_qkv and p.ndim == 2 and g.size(0) == 3 * g.size(1):
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
                nuc_norm_approx = nuclear_norm_via_newtonschulz5(
                    g.reshape(len(g), -1), steps=6, eps=eps
                )
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
            nuc_norm_approx = nuclear_norm_via_newtonschulz5(
                g.reshape(len(g), -1), steps=6, eps=eps
            )
            nuc_norm_approx = nuc_norm_approx / (min(g.shape[-2], g.shape[-1]))
            effective_lr *= nuc_norm_approx.clamp(min=0.1)

        if weight_decay > 0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

        p.data.add_(update, alpha=-effective_lr)

    def _adam_step(self, p, g, group, state):
        lr           = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps          = group["adam_eps"]

        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        exp_avg    = state["exp_avg"]
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
        lr           = group["lr"]
        eps          = group["eps"]
        momentum     = group["momentum"]
        nesterov     = group["nesterov"]
        weight_decay = group["weight_decay"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)

        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        g_eff = g.add(buf, alpha=momentum) if nesterov else buf

        # steepest descent under the \ell1->RMS operator norm:
        # normalize each column by its RMS
        col_rms = torch.sqrt((g_eff ** 2).mean(dim=0, keepdim=True) + eps)  # (1, vocab)
        # divide by sqrt of input dimension too
        update = g_eff / (col_rms * g_eff.shape[0])  # (embed_dim, vocab) — each col has unit RMS

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
            embedding_layer_optim = group.get("embedding_layer_optim", "muon")

            for p in group["params"]:
                if p.grad is None:
                    continue

                g        = p.grad
                state    = self.state[p]
                opt_type = self._optimizer_types.get(id(p), "muon")

                if opt_type == "embed":
                    if embedding_layer_optim == "muon":
                        self._muon_step(p, g, group, state)
                    elif embedding_layer_optim == "adam":
                        self._adam_step(p, g, group, state)
                    else:  # "rms"
                        self._embedding_layer_step(p, g, group, state)
                elif opt_type == "muon":
                    self._muon_step(p, g, group, state)
                else:
                    self._adam_step(p, g, group, state)