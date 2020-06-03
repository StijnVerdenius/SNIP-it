from models.criterions.CNIP import CNIP


class CNIPit(CNIP):

    """
    Original creation from our paper: https://arxiv.org/abs/2006.00896
    Combines SNIP (unstructured) and SNAP (structured) on equal footing in an itertive algorithm
    Implements CNIP-it (before training)
    """

    def __init__(self, *args, limit=0.0, start=0.5, steps=5, **kwargs):
        self.limit = limit
        super(CNIPit, self).__init__(*args, **kwargs)

        # define the k_i percentages
        self.steps = [limit - (limit - start) * (0.5 ** i) for i in range(steps + 1)] + [limit]

        # working variables
        self.left = 1.0
        self.pruned = 0.0

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, *args, **kwargs):

        # iteratively prune with CNIP
        while len(self.steps) > 0:

            # get planned k_i
            percentage = self.steps.pop(0)
            prune_now = (percentage - self.pruned) / (self.left + 1e-8)

            # prune
            super().prune(percentage=prune_now, *args, **kwargs)

            # determine how much actually got pruned
            self.pruned = self.model.pruned_percentage  # percentage
            self.left = 1.0 - self.pruned
