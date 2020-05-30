import torch
import torch.nn as nn

from models.Pruneable import Pruneable
from utils.constants import PROD_BIG_POOL, BIG_POOL, PROD_MIDDLE_POOL, MIDDLE_POOL


class Conv6(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1,), **kwargs):
        super(Conv6, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        channels, _, _ = input_dim

        leak = 0.05
        gain = nn.init.calculate_gain('leaky_relu', leak)

        self.conv = nn.Sequential(

            self.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1, bias=True, gain=gain),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            self.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, gain=gain),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=2),

            self.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True, gain=gain),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            self.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True, gain=gain),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(kernel_size=2),

            self.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True, gain=gain),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            self.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True, gain=gain),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),
            # nn.MaxPool2d(kernel_size=2),

        ).to(device)

        self.avgpool = nn.AdaptiveAvgPool2d(MIDDLE_POOL).to(device)

        self.fc = nn.Sequential(

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(256*PROD_MIDDLE_POOL, 256, bias=True, gain=gain),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(256, 256, bias=True, gain=gain),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(leak),

            nn.Dropout(p=0.3, inplace=False),
            self.Linear(256, output_dim, bias=True),

        ).to(device)

    def forward(self, x: torch.Tensor):
        x = self.conv.forward(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = Conv6(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)
