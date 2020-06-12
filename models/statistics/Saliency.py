import torch


class Saliency:

    """
    Analyses saliency maps
    Loosly based on the paper:
    Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    https://arxiv.org/abs/1312.6034
    and the public python library:
    flashtorch
    https://github.com/MisaOgura/flashtorch
    """

    def __init__(self, model, device, x):

        self.model = model
        self.device = device
        x: torch.Tensor
        self.x = x.to(device).float()

    def get_grad(self):
        self.model.eval()

        self.x.requires_grad = True

        self.gradients = torch.zeros(self.x.shape)

        self.model.zero_grad()

        self.gradients = torch.zeros(self.x.shape)

        output = self.model.forward(self.x)

        _, top_class = output.topk(1, dim=1)

        target = torch.FloatTensor(output.shape).zero_().to(self.device)

        target[[x for x in range(len(output))], top_class.t().flatten()] = 1

        output.backward(gradient=target)

        gradients = self.x.grad

        images = self._get_images(gradients).mean(dim=0)
        images = images.unsqueeze(dim=0)

        self.x.requires_grad = False
        self.x.grad = None

        self.model.train()

        return images

    def _get_images(self, gradients):
        with torch.no_grad():
            grads = self._process_images(gradients.detach().cpu())
            originals = self._process_images(self.x.detach().cpu())
            sh = grads.shape
            final = torch.zeros((sh[1], sh[2] * 8, sh[3] * 2))

            for j in range(2):
                for i in range(grads.shape[0]):
                    final[:, i * sh[2]:sh[2] * (i + 1), 0:sh[3]] = grads[i]
                    final[:, i * sh[2]:sh[2] * (i + 1), sh[3]:2 * sh[3]] = originals[i]

            final = torch.cat([final[:, :sh[2] * 4, :], final[:, sh[2] * 4:, :]], dim=2)

            return final

    def _process_images(self, tensor):

        min_value = 0.0
        max_value = 1.0
        saturation = 0.1
        brightness = 0.5

        mean = tensor.mean()
        std = tensor.std()

        if std == 0:
            std += 1e-7

        standardized = tensor.sub(mean).div(std).mul(saturation)

        return standardized.add(brightness).clamp(min_value, max_value)
