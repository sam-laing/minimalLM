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
    

def make_eps_func(
        e_0: float = 1e-8, 
        e_max: float = 1e-3, 
        w: float = 0.2, 
        E: float = 0.8, 
        T: int = 1000, 
        kappa: float = 0.9  # Controls the growth rate
    ):
    def eps_func(t):
        assert (t >= 0 and t <= T), "t must be in [0, T]"
        assert kappa > 0, "kappa must be positive"

        # Define k based on the transition interval length
        L = T * (E - w)  # Length of the transition interval
        k = kappa / L    # Scale k with the interval length

        if t >= 0 and t < w * T:
            return e_0
        elif t >= E * T and t <= T:
            return e_max
        elif t >= w * T and t <= E * T:
            # Exponential transition
            A = (e_max - e_0) / (math.exp(kappa) - 1)
            B = e_0 - A
            return A * math.exp(k * (t - w * T)) + B

    return eps_func

class Adam2SGD(EpsScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        step_size: int,
        max_eps: float, 
        init_eps: float = 1e-8,
        transition_function: str = "exponential", 
        growth_factor: float = 10.0,
        last_epoch: int = -1,
        verbose: bool = False, 
        warmup: float = 0.3, 
        end_phase: float = 0.85,
        total_steps: int = 1000
    ):
        self.step_size = step_size
        self.warmup_steps = warmup * total_steps
        self.end_steps = end_phase * total_steps
        self.transition_function = transition_function
        

        self.eps_func = make_eps_func(e_0=init_eps, e_max=max_eps, w=warmup, E=end_phase, T=total_steps, kappa=0.9)

        super().__init__(optimizer, last_epoch, verbose)

    def get_eps(self) -> List[float]:
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['eps'] for group in self.optimizer.param_groups]
        return [self.eps_func(self.last_epoch) for group in self.optimizer.param_groups]
