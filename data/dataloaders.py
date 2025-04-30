
import torch

from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from data.datasamplers import StatefulSequentialSampler, StatefulDistributedSampler


def get_dataloaders(cfg, start_step: int = 0):
  """Load trainset and perhaos validset. Returns correspondent DataLoaders."""
  
  train_set = load_from_disk(cfg.trainset_path)
  if not isinstance(train_set , Dataset):
    raise ValueError("dataset should be a datasets.Dataset")
  
  sampler = _get_sampler(train_set, cfg, start_step)
  
  trainloader = DataLoader(
    train_set,
    sampler = sampler,
    batch_size = cfg.micro_batch_size,
    num_workers = cfg.num_workers,
    pin_memory = True,
    # drop_last = True, # raise error when True and batch_size is specified
    prefetch_factor = 2 if cfg.num_workers > 0 else None,
    persistent_workers = True if cfg.num_workers > 0 else False,
  )

  if not cfg.validset_path:
    validloader = None
  else:
    valid_set = load_from_disk(cfg.validset_path)
    if not isinstance(valid_set, Dataset):
      raise ValueError("'dataset' should be a datasets.Dataset")

    if dist.is_initialized():
      valid_sampler = DistributedSampler(valid_set, drop_last=True)
    else:
      valid_sampler = SequentialSampler(valid_set)
      
    validloader = DataLoader(
      valid_set,
      batch_size = cfg.micro_batch_size,
      num_workers = cfg.num_workers,
      sampler = valid_sampler,
    )
  
  return trainloader, validloader


def _get_sampler(train_set, cfg, start_step):
  """Initlaizes a sampler for a torch.Dataloader.
  Options:
    - sequential sampler
    - random sampler
  We implement "stateful" sequential samplers for resuming training from a specified step.
  """  
  ddp = dist.is_initialized()
  
  # Determine which seed to use
  seed_to_use = cfg.seed if cfg.one_seeded else cfg.sampler_seed
  
  if cfg.resume:
    if ddp:
      sampler = StatefulDistributedSampler(train_set, batch_size=cfg.micro_batch_size, seed=seed_to_use)
      sampler.set_start_iter(start_step)
    else:
      sampler = StatefulSequentialSampler(train_set, batch_size=cfg.micro_batch_size, start_idx=start_step, seed=seed_to_use)
  else:
    if ddp:
      # For non-resumed DDP, also use the unified seed if one_seeded
      sampler = DistributedSampler(train_set, drop_last=True, seed=seed_to_use if seed_to_use is not None else 0)
    else:
      sampler = SequentialSampler(train_set)  # equivalent to StatefulSequentialSampler(..., start_idx=0)
  
  if cfg.sampler == 'random' and not ddp:
    # For random sampling, use the same unified seed approach
    generator = torch.Generator().manual_seed(seed_to_use)
    sampler = RandomSampler(train_set, generator=generator)
  
  return sampler