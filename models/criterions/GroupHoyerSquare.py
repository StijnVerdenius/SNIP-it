from random import randint

import torch

from models.criterions.StructuredRandom import StructuredRandom
from utils.constants import HOYER_THERSHOLD


class GroupHoyerSquare(StructuredRandom):

    """
    Our interpretation/implementation of the threshold trimming part (only) of Group-HS from the paper:
    DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant Sparsity Measures
    https://arxiv.org/abs/1908.09979
    """

    def __init__(self, *args, **kwargs):
        super(GroupHoyerSquare, self).__init__(*args, **kwargs)
        self.pruned = False

    def prune(self, *args, **kwargs):
        if self.pruned:
            return
        else:
            self.pruned = True
            super().prune(*args, **kwargs)

    def get_out_vector(self, param, dims):
        """ returns a vector which determines which nodes from the output dimension to keep """


        # threshold trimming
        prune_dim = [1] + ([] if len(param.shape) <= 2 else [2, 3])
        return (param.abs() > HOYER_THERSHOLD).sum(dim=tuple(prune_dim)) > 0  # dims[1]

    def get_in_vector(self, param, dims):
        """ returns a vector which determines which nodes from the input dimension to keep """

        # threshold trimming
        prune_dim = [0] + ([] if len(param.shape) <= 2 else [2, 3])
        return (param.abs() > HOYER_THERSHOLD).sum(dim=tuple(prune_dim)) > 0  # dims[0]

    def get_inclusion_vectors(self, i, in_indices, last_is_conv, module, modules, out_indices, percentage):
        """ returns a vectors which determine which nodes to keep """

        param = module.weight
        dims = param.shape[:2]  # out, in
        if in_indices is None:
            in_indices = torch.ones(dims[1])
            if self.model._outer_layer_pruning:
                in_indices = self.get_in_vector(param, dims)
        else:
            in_indices = out_indices
        out_indices = self.get_out_vector(param, dims)
        is_last = (len(modules) - 1) == i
        if is_last and not self.model._outer_layer_pruning:
            out_indices = torch.ones(dims[0])
        now_is_fc = isinstance(module, torch.nn.Linear)
        now_is_conv = isinstance(module, torch.nn.Conv2d)
        if last_is_conv and now_is_fc:
            ratio = param.shape[1] // in_indices.shape[0]
            in_indices = torch.repeat_interleave(in_indices, ratio)
        last_is_conv = now_is_conv
        if in_indices.sum() == 0:
            in_indices[randint(0, len(in_indices) - 1)] = 1
        if out_indices.sum() == 0:
            out_indices[randint(0, len(out_indices) - 1)] = 1
        return in_indices.bool(), last_is_conv, now_is_conv, out_indices.bool()
