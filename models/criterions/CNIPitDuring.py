from models.criterions.CNIP import CNIP
from models.criterions.CNIPit import CNIPit


class CNIPitDuring(CNIPit):

    """
    Original creation from our paper: todo
    Combines SNIP (unstructured) and SNAP (structured) on equal footing in an itertive algorithm
    Implements CNIP-it (during training)
    """

    def __init__(self, *args, **kwargs):
        super(CNIPitDuring, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        # get planned k_i
        percentage = self.steps.pop(0)
        prune_now = (percentage - self.pruned) / (self.left + 1e-8)

        # prune
        kwargs["percentage"] = prune_now
        CNIP.prune(self, **kwargs)

        # determine what was pruned
        self.pruned = self.model.structural_sparsity  # use structured percentage
        self.left = 1.0 - self.pruned
