import torch

from models.trainers.DefaultTrainer import DefaultTrainer


class L0Trainer(DefaultTrainer):

    """
    inherits from defauttrainer to handle l0-regularisation
    """

    def validate_wrapper(self, func):
        def wrapper():
            old_params = None
            if self._model.beta_ema > 0:
                self._model.eval()
                old_params = self._model.get_params
                self._model.load_ema_params()
            out = func()
            if self._model.beta_ema > 0:
                self._model.eval()
                self._model.load_params(old_params.values())
                self._model.train()
            return out

        return wrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate = self.validate_wrapper(self.validate)

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # update metrics
        self._metrics.update_batch(train)

        # record time
        if "cuda" in str(self._device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)

        # backward pass
        if train:
            self._backward_pass(loss)

            # clamp the parameters
            layers = [module for name, module in self._model.named_modules() if "L0" in name]
            for layer in layers:
                layer.constrain_parameters()

            if self._model.beta_ema > 0.:
                self._model.update_ema()

                # record time
        if "cuda" in str(self._device):
            end.record()
            torch.cuda.synchronize(self._device)
            time = start.elapsed_time(end)
        else:
            time = 0

        # free memory
        for tens in [out, y, x, loss]:
            tens.detach()

        return accuracy, loss.item(), time

    def _add_metrics(self, *args):
        """
        save metrics
        """

        super()._add_metrics(*args)
        self._metrics.add(int(self._model.expected_l0), key="norm/l0")
