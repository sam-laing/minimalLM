
import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from models import get_param_groups
from optim import intialize_optimizer, initalize_scheduler


def _build_orthogonal_matrix(n, device, dtype, seed):
  """Haar-random n x n orthogonal matrix via QR of a Gaussian, fixed by seed."""
  gen = torch.Generator(device='cpu').manual_seed(seed)
  A = torch.randn(n, n, generator=gen, dtype=torch.float32)
  Q, R_diag = torch.linalg.qr(A)
  # fix sign ambiguity of QR so Q is uniformly Haar-distributed
  Q = Q * torch.sign(torch.diagonal(R_diag)).unsqueeze(0)
  return Q.to(device=device, dtype=dtype)


def _move_to_device(batch, seq_len, device):
  """Slice batch to get inputs and targets, and move them to device."""
  
  inputs = batch['input_ids'][:,:seq_len]
  targets = batch['input_ids'][:,1:(seq_len+1)]

  if 'cuda' in device:
    # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.pin_memory().to(device, non_blocking=True)
  else:
    inputs, targets = inputs.to(device), targets.to(device)

  return inputs, targets


class TorchEngine(torch.nn.Module):
  """
  A module containing model, optimizer, scheduler, grad scaler.
  Wraps together a training step. Takes care of grad accumulation.
  """
  def __init__(
      self,
      model,
      cfg,
      device,
      local_rank,
      ckpt,
      ):
    super().__init__()
    
    self.micro_steps = 0
    self.accumulated_samples = 0

    self.seq_len = cfg.seq_len
    self.accumulation_steps = cfg.grad_accumulation_steps
    self.grad_clip = cfg.grad_clip
    self.dtype = cfg.dtype

    self.device = device
    
    # Load model state dict
    if cfg.resume:
      model.load_state_dict(ckpt['state_dict'])
      self.micro_steps = ckpt['micro_step']

    # Move model to device and to DDP
    self.model = model.to(device)
    if torch.distributed.is_initialized():
      self.model = DDP(self.model, device_ids=[local_rank])

    # Compile
    if cfg.torch_compile:
      print(f"Compiling the model...")
      self.model = torch.compile(self.model)

    # AMP
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
    self.scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))

    # Loss
    self.criterion = CrossEntropyLoss()

    # Optimizer
    param_groups = get_param_groups(model, cfg.weight_decay)
    self.optimizer = intialize_optimizer(param_groups, cfg)
    self.scheduler = initalize_scheduler(self.optimizer, cfg)

    if cfg.resume:
      self.optimizer.load_state_dict(ckpt['optimizer'])
      self.scheduler.load_state_dict(ckpt['scheduler'])
      self.scaler.load_state_dict(ckpt['scaler'])

    # Rotation-robustness ablation: fix a random orthogonal R once, and wrap
    # the embedding's optimizer step so the *effective* weight update is
    # computed in a rotated basis and then rotated back. For an optimizer
    # whose update rule is equivariant to G -> G @ R (Muon's orthogonalize
    # step provably is), this is a no-op up to floating point. For one that
    # isn't (e.g. Adam's elementwise second-moment estimate), it isn't.
    self.rotate_embedding = getattr(cfg, 'rotate_embedding', False)
    self.embed_param = None
    self.rotation_R = None
    if self.rotate_embedding:
      named_params = dict(self.model.named_parameters())
      embed_name = next(n for n in named_params if n.endswith('embed_tokens.weight'))
      self.embed_param = named_params[embed_name]
      d_model = self.embed_param.shape[-1]
      seed = getattr(cfg, 'rotation_seed', 0)
      self.rotation_R = _build_orthogonal_matrix(d_model, device, self.embed_param.dtype, seed)
      print(f"Rotation-robustness ablation ON: rotating '{embed_name}' "
            f"(seed={seed}, R is {d_model}x{d_model})")


  def step(self, batch):
    """Wraps a fwd pass, backwd pass, and optimization step."""
    
    self.model.train()
    
    self.micro_steps += 1
    self.accumulated_samples += 1

    inputs, targets = _move_to_device(batch, self.seq_len, self.device)

    # sync (reduce) gradients at the last accumulation step
    if torch.distributed.is_initialized():
      self.model.require_backward_grad_sync = \
        (self.accumulated_samples == self.accumulation_steps)

    # forward pass with autocasting
    with self.ctx:
      output = self.model(inputs)
      logits = getattr(output, 'logits', output)
      loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
      loss = loss / self.accumulation_steps

    # detach for logging (scale up to undo the division above)
    loss_val = loss.detach() * self.accumulation_steps
    if loss>1e6 or torch.isnan(loss): 
      raise ValueError("Train loss is nan")

    # backward pass, with gradient scaling if training in fp16
    self.scaler.scale(loss).backward()

    # step after accumulation
    if self.accumulated_samples == self.accumulation_steps:
      self.accumulated_samples = 0

      if self.grad_clip:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

      # rotation-robustness ablation: rotate the embedding's gradient before
      # the optimizer sees it (R fixed at init, same R every step)
      embed_true_grad, embed_w_before = None, None
      if self.rotate_embedding and self.embed_param.grad is not None:
        embed_true_grad = self.embed_param.grad.clone()
        embed_w_before = self.embed_param.data.clone()
        self.embed_param.grad.copy_(embed_true_grad @ self.rotation_R)

      # step the optimizer, step the scaler if training in fp16
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # un-rotate the *effective update* the optimizer just applied, and
      # restore the true (pre-rotation) grad so checkpointing below sees
      # what the model actually produced, not the internal rotated copy
      if embed_w_before is not None:
        delta_rotated = self.embed_param.data - embed_w_before
        self.embed_param.data.copy_(embed_w_before + delta_rotated @ self.rotation_R.T)
        self.embed_param.grad.copy_(embed_true_grad)

      # stash grads at this iterate for checkpointing, before they're flushed
      self.last_grads = {
        name: p.grad.detach().clone()
        for name, p in self.model.named_parameters() if p.grad is not None
      }

      # flush the gradients
      self.optimizer.zero_grad(set_to_none=True)
      
      # step the scheduler
      if self.scheduler:
        self.scheduler.step()
  
    return loss_val


  @torch.no_grad()
  def eval(self, dataloader):
    """Evaluate model on a dataloader."""
    
    self.model.eval()
    
    # Compute loss on dataloader
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
      inputs, targets = _move_to_device(batch, self.seq_len, self.device)
      with self.ctx:
        output = self.model(inputs)
        logits = getattr(output, 'logits', output)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    
      if torch.isnan(loss) or loss is None:
        raise ValueError("Validation loss is nan")
    
      total_loss += loss.item()
      num_batches += 1

    # reduce loss across processes
    if dist.is_initialized():
      total_loss_tensor = torch.tensor([total_loss], device=self.device)
      num_batches_tensor = torch.tensor([num_batches], device=self.device, dtype=torch.int)
      dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
      total_loss = total_loss_tensor.item() / dist.get_world_size()
      num_batches = num_batches_tensor.item() // dist.get_world_size() # superflous if drop_last=True in dataloader

    # calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss
