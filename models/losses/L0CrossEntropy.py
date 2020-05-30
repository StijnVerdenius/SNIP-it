from models.GeneralModel import GeneralModel
from models.losses.CrossEntropy import CrossEntropy


class L0CrossEntropy(GeneralModel):

    """
    adds l0 to crossentropy
    """

    def __init__(self, device, *args, l0_reg=1, **kwargs):
        super(L0CrossEntropy, self).__init__(device, **kwargs)
        self.l0_reg = l0_reg
        self.likelihood = CrossEntropy(device, *args, **kwargs)

    def forward(self, output=None, model=None, target=None, weight_generator=None, **kwargs):
        log_likes = self.likelihood.forward(output=output, target=target, weight_generator=weight_generator, **kwargs)

        l0 = model.l0_regularisation

        return log_likes + l0
