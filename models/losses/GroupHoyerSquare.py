import torch

from models.losses.CrossEntropy import CrossEntropy


class GroupHoyerSquare(CrossEntropy):

    """
    Adapted code for Group-HS of the paper:
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
                if param.requires_grad and len(list(param.size())) == 4 and torch.sum(torch.abs(param)) > 0:

                    reg += ((torch.sum(torch.sqrt(torch.sum(param ** 2, (0, 2, 3)))) ** 2) + (
                            torch.sum(torch.sqrt(torch.sum(param ** 2, (1, 2, 3)))) ** 2)) / torch.sum(param ** 2)

                elif param.requires_grad and len(list(param.size())) == 2 and torch.sum(torch.abs(param)) > 0:

                    reg += ((torch.sum(torch.sqrt(torch.sum(param ** 2, 0))) ** 2) + (
                            torch.sum(torch.sqrt(torch.sum(param ** 2, 1))) ** 2)) / torch.sum(param ** 2)

                else:
                    raise Exception("uncommon shape for parameter:", param.shape)
        return super().forward(weight_generator=weight_generator, **kwargs) + (reg * self.hoyer_reg)
