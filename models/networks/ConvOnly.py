import torch
import torch.nn as nn

from models.Pruneable import Pruneable


class ConvOnly(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1,), **kwargs):
        super(ConvOnly, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        channels, _, _ = input_dim

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        self.features = nn.Sequential(
            self.Conv2d(channels, 64, kernel_size=5, padding=1, gain=gain),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=2),
            self.Conv2d(64, 128, kernel_size=3, padding=1, gain=gain),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=2),
            self.Conv2d(128, 64, kernel_size=3, padding=1, gain=gain),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=2),
            self.Conv2d(64, 32, kernel_size=3, padding=1, gain=gain),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=3),
            self.Conv2d(32, output_dim, kernel_size=3, padding=1),
        ).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

    def forward(self, x: torch.Tensor, **kwargs):
        return self.avgpool.forward(self.features.forward(x, **kwargs)).squeeze()


if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = ConvOnly(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)
