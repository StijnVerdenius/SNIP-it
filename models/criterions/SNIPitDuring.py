from models.criterions.SNIP import SNIP
from models.criterions.SNIPit import SNIPit


class SNIPitDuring(SNIPit):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNIP-it (during training)
    """

    def __init__(self, *args, **kwargs):
        super(SNIPitDuring, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) > 0:
            # determine k_i
            percentage = self.steps.pop(0)
            kwargs["percentage"] = percentage

            # prune
            SNIP.prune(self, **kwargs)
