from torch.optim import Optimizer
class EpsilonScheduler:
    """   
    alternative to learning rate scheduler in later epochs
    idea is that v_k becomes smaller, this will cause an explosion in step size
    epslion parameter is traditionally used to prevent division by zero but here 
    we use it to somewhat regularize the step size
    """

    def __init__(self, optimizer: Optimizer, milestones, epochs, eps:float = 1e-8):
        assert isinstance(optimizer, Optimizer), "optimizer should be an instance of torch.optim.Optimizer"

        self.optimizer = optimizer
        self.eps = eps
        self.epoch = -1

    def step(self):
        self.epoch +=1