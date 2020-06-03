import torch

from models.criterions.SNAP import SNAP
from models.criterions.SNIP import SNIP


class CNIP(SNIP, SNAP):

    """
    Original creation from our paper: https://arxiv.org/abs/2006.00896
    Combines SNIP (unstructured) and SNAP (structured) on equal footing
    """

    def __init__(self, *args, **kwargs):
        super(CNIP, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, **kwargs):

        # get weigt elasticity
        x = SNIP.get_weight_saliencies(self, train_loader)
        weight_scores = x[0]

        # get node elasticity
        y = SNAP.get_weight_saliencies(self, train_loader)
        node_scores = y[0]

        # combined
        all_scores = torch.cat([weight_scores, node_scores])

        # get threshold
        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        if num_params_to_keep < 1:
            num_params_to_keep += 1
        elif num_params_to_keep > len(all_scores):
            num_params_to_keep = len(all_scores)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        # get percentages
        percentage_weights = (weight_scores < acceptable_score).sum().item() / len(weight_scores)
        percentage_nodes = (node_scores < acceptable_score).sum().item() / len(node_scores)

        print("fraction for pruning nodes", percentage_nodes, "fraction for pruning weights", percentage_weights)

        # prune
        SNIP.handle_pruning(self, weight_scores, x[1], x[2], manager, x[-1], percentage_weights)
        SNAP.handle_pruning(self, node_scores, y[1], y[3], percentage_nodes)
