import torch
import time

# Simulate your model's matrix sizes
shapes = [
    (2304, 768),  # w_qkv
    (768, 768),   # w_out
    (2048, 768),  # GLU fc1
    (768, 2048),  # GLU fc2
]

device = torch.device('cuda')
n_iters = 100

for shape in shapes:
    G = torch.randn(shape, device=device, dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        _ = torch.linalg.svd(G.float(), full_matrices=False)
    torch.cuda.synchronize()
    
    # Time SVD
    start = time.time()
    for _ in range(n_iters):
        U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
        _ = U @ Vh
    torch.cuda.synchronize()
    svd_time = (time.time() - start) / n_iters * 1000
    
    # Time Newton-Schulz (simplified, non-compiled)
    start = time.time()
    for _ in range(n_iters):
        X = G.bfloat16() / (G.norm() + 1e-7)
        if X.size(0) > X.size(1):
            X = X.T
        for _ in range(5):
            A = X @ X.T
            X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    torch.cuda.synchronize()
    ns_time = (time.time() - start) / n_iters * 1000
    
    print(f"{shape}: SVD={svd_time:.2f}ms, NS={ns_time:.2f}ms, ratio={svd_time/ns_time:.1f}x")