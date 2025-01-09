from torch.optim import Optimizer
class EpsilonScheduler:
    def __init__(self, optimizer: Optimizer, milestones, epochs, eps = 1e-8):
        self.optimizer = optimizer
        self.eps = eps
        self.last_epoch = -1

    def step(self):
        self.last_epoch +=1