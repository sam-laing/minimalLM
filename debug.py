from models import Transformer

from dataclasses import dataclass
from typing import Optional, Union, Literal

@dataclass
class ModelConfig:
    # Basic settings
    deterministic: bool = True
    seed: int = 17
    one_seeded: bool = True
    
    # Data paths and settings
    trainset_path: str = "/fast/slaing/data/lm/slim_pajama/new_sp_15B_tokens/train"
    vocab_size: int = 50280
    seq_len: int = 2048
    sampler: str = 'sequential'
    sampler_seed: Optional[int] = None
    num_workers: int = 4
    
    # Evaluation settings
    eval: bool = True
    validset_path: str = "/fast/slaing/data/lm/slim_pajama/new_valid/validation"
    eval_every_steps: int = 99999999999
    eval_final: bool = True
    
    # Model architecture
    model: str = 'transformer'
    d_model: int = 768
    mlp_class: str = 'glu'
    expand: str = '8/3'
    n_layers: int = 12
    n_heads: int = 12
    rms_norm: bool = True
    tie_embeddings: bool = False
    torch_compile: bool = True
    
    # Training settings
    steps_budget: int = 6200
    micro_batch_size: int = 32
    grad_accumulation_steps: int = 8
    dtype: str = 'bfloat16'
    
    # Optimizer settings
    optim: str = "muon"
    fused_optim: bool = False
    lr: float = 0.003  
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    beta1: float = 0.925 
    beta2: float = 0.95
    dampening: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 1.0
    do_bias_correction: bool = False
    zero_init: bool = True
    eps_inside_sqrt: bool = False
    adam_to_sgd_ratio: float = 1.0
    
    # Scheduler settings
    scheduler: str = "warmup_cosine"
    warmup_steps: float = 0.1
    cooldown_steps: Optional[float] = None
    lr_start: float = 1e-6
    lr_end: float = 1e-6
    lr_end_pct: Optional[float] = None
    
    # Logging and output
    log_every_steps: int = 1
    print_progress: bool = True
    use_wandb: bool = True
    wandb_project: str = 'llm_bias_correction'
    wandb_dir: str = '/fast/slaing/logs/llm/wandb'
    wandb_run_name: str = "bias_correction_test"
    exp_name: str = 'tr_check_not_fused'
    out_dir: str = '/fast/slaing/exp/llm'
    wandb_api_key: str = "0b4cb5f85aad923fbaec987675544b3d4530b0e4"
    over_write: bool = True
    
    # Resume settings
    resume: bool = False
    resume_micro_step: Optional[int] = None
    resume_exp_name: Optional[str] = None
    
    # Checkpoint settings
    save_last_checkpoint: bool = True
    save_intermediate_checkpoints: bool = False
    save_every_steps: Optional[int] = None
    
    # Analysis settings
    track_moments: bool = False
    moment_plots: str = '/fast/slaing/adam_moment_plots/llm'

from models import construct_model  

import torch  
from optim.muon import Muon
def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"


if __name__ == "__main__":
    cfg = ModelConfig()

    
    # Construct the model
    model, model_cfg = construct_model(cfg)

    #iterate through layers and print their names and shapes
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}")
    print(f"Model configuration: {model_cfg}")
    from optim.init_optim import intialize_optimizer

    optimizer = intialize_optimizer(model.parameters(), cfg)
    print(f"Optimizer: {optimizer.__class__.__name__}")

    #some bullshit data and do a forward pass and optimizer step 
    bsz = 2
    seq_len = 2048
    x = torch.randint(0, model_cfg.vocab_size, (bsz, seq_len)).long()
    x = x.cuda()  # Move to GPU

    model = model.cuda()  
    optimizer.zero_grad()
    output = model(x)  
   
    target = torch.randint(0, model_cfg.vocab_size, (bsz, seq_len)).long().cuda()  
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, model_cfg.vocab_size), target.view(-1))  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Optimizer step
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    print("Forward pass and optimizer step completed successfully.")
