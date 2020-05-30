from models.criterions.SNAP import SNAP
from models.criterions.SNAPit import SNAPit


class SNAPitDuring(SNAPit):

    """
    Original creation from our paper: todo
    Implements SNAP-it (during training)
    """

    def __init__(self, *args, **kwargs):
        super(SNAPitDuring, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) == 0:
            print("finished all pruning events already")
            return

        # get k_i
        percentage = self.steps.pop(0)
        prune_now = (percentage - self.pruned) / (self.left + 1e-8)

        # prune
        kwargs["percentage"] = prune_now
        SNAP.prune(self, **kwargs)

        # adjust
        self.pruned = self.model.structural_sparsity  # percentage
        self.left = 1.0 - self.pruned
