import torch

from models.criterions.General import General
from utils.constants import HOYER_THERSHOLD


class HoyerSquare(General):

    """
    Own interpretation of the threshold trimming part (only) of HoyerSquare from the paper:
    DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant Sparsity Measures
    https://arxiv.org/abs/1908.09979
    """

    def __init__(self, *args, **kwargs):
        super(HoyerSquare, self).__init__(*args, **kwargs)
        self.pruned = False

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):

        # only prune once
        if self.pruned:
            return
        else:
            self.pruned = True

            # threshold trimming
            for (name, weights) in self.model.named_parameters():

                if name in self.model.mask:
                    mask = weights.abs() > HOYER_THERSHOLD

                    self.model.mask[name] = mask

                    mask: torch.Tensor

                    print("Sparsity layer", name, ":", ((mask == 0).sum().float() / float(mask.numel())).item())

            self.model.apply_weight_mask()
            print("sparsity after pruning", self.model.pruned_percentage)
