import torch

from models.losses.CrossEntropy import CrossEntropy


class HoyerSquare(CrossEntropy):

    """
    Adapted code for HoyerSquare of the paper:
    DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant Sparsity Measures
    https://arxiv.org/abs/1908.09979
    Sourced form their github at:
    https://github.com/yanghr/DeepHoyer
    """

    def __init__(self, device, *args, hoyer_reg=0, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.hoyer_reg = hoyer_reg

    def forward(self, *args, weight_generator=None, criterion=None, **kwargs):
        reg = 0.0
        if not criterion.pruned:
            for param in weight_generator:
                if len(param.shape) < 2:
                    continue
                reg += (torch.sum(torch.abs(param)) ** 2) / torch.sum(param ** 2)
        return super().forward(weight_generator=weight_generator, **kwargs) + (reg * self.hoyer_reg)
