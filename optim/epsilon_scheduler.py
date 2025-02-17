from typing import List, Optional
import torch
from torch.optim import Optimizer
from optim.nestingMA import NestedMA

class EpsScheduler:
    """Base class for epsilon parameter schedulers."""
    
    def __init__(
            self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
            ):
        assert isinstance(optimizer, Optimizer), ""
        # Validate optimizer has eps parameter
        if not (isinstance(optimizer, torch.optim.AdamW) or isinstance(optimizer, NestedMA)): 
            raise ValueError("Optimizer must be AdamW or similar with eps parameter")
            
        self.optimizer = optimizer
        self.verbose = verbose
        self.base_eps = []  # Store initial epsilon values
        
        for group in optimizer.param_groups:
            self.base_eps.append(group['eps'])
            
        self.last_epoch = last_epoch
        
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_eps', group['eps'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_eps' not in group:
                    group.setdefault('initial_eps', self.base_eps[i])
                    
        self.step()
    
    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a dict."""
        return {
            'base_eps': self.base_eps,
            'last_epoch': self.last_epoch
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the scheduler state."""
        self.base_eps = state_dict['base_eps']
        self.last_epoch = state_dict['last_epoch']
    
    def get_eps(self) -> List[float]:
        """Compute current eps value. Override in subclasses."""
        raise NotImplementedError
        
    def step(self) -> None:
        """Update epsilon parameter in optimizer."""
        self.last_epoch += 1
        values = self.get_eps()
        
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, eps = data
            param_group['eps'] = eps
            if self.verbose:
                print(f'Adjusting eps to {eps} for parameter group {i}')

class StepEpsScheduler(EpsScheduler):
    """Step scheduler that increases epsilon by growth_factor every step_size epochs."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        growth_factor: float = 2.0, 
        last_epoch: int = -1,
        verbose: bool = False
    ):
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0 for increasing epsilon, got {growth_factor}")
            
        self.step_size = step_size
        self.growth_factor = growth_factor  # Renamed from gamma
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_eps(self) -> List[float]:
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['eps'] for group in self.optimizer.param_groups]
        return [group['eps'] * self.growth_factor for group in self.optimizer.param_groups]