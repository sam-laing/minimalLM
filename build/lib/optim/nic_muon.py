""" "
Muon torch implementations.
"""

import torch
import torch.distributed as dist
from abc import ABC, abstractmethod


@torch.compile()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to approximally orthogonalize G.
    5-th order odd polynomial to approximate sign(x) on [-1,1],
    pushing singlular values to {+1,-1}.

    M = U @ S @ V.T
    sign(M) = U @ sign(S) @ V.T, odd matrix polynomial commutes with SVD
    sign(x) ~= a*x + b*x^3 + c*x^5, x in [-1,1]
    """
    if G.ndim != 2:
        raise RuntimeError(f"Expected 2D tensor in N-S, found {G.ndim} instead.")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1.
    # Ortho(cX)=Ortho(X), so we can normalize by ||X||_2 <= ||X||_F
    X /= X.norm() + eps

    # NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuonBase(torch.optim.Optimizer, ABC):
    """Muon optimizer - Momentum Orthogonalized by Newton-Schulz.

    Abstract class.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        beta=0.95,
        nesterov=True,
        ns_steps=5,
        ns_eps=1.0e-7,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid muon_beta parameter: {beta}")
        if nesterov not in [True, False]:
            raise ValueError(f"Invalid nesterov parameter: {nesterov}")
        if not 0 < ns_steps:
            raise ValueError(f"Invalid ns_steps parameter: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps parameter: {ns_eps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )
        super().__init__(params, defaults)

    @abstractmethod
    @torch.no_grad()
    def step(self, closure=None):
        pass


class MuonVanilla(MuonBase):
    """
    Single Devide implementation: if used with DDP,
    it will replicate computation across devices.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["beta"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            ns_eps = group["ns_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                state["m"].mul_(beta).add_(g, alpha=1 - beta)

                if nesterov:
                    g = g.add(state["m"], alpha=beta)
                else:
                    g = state["m"]
                g = g.reshape(g.size(0), -1)  # flatten trailing dims (3D, 4D)
                g = zeropower_via_newtonschulz5(g, steps=ns_steps, eps=ns_eps)
                g = g.view(p.shape)  # restore original shape
                # Adjust from spectral norm 1 to RMS operator norm 1 https://arxiv.org/abs/2310.17813
                g *= max(1.0, p.size(-2) / p.size(-1)) ** 0.5

                p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)

        return loss


