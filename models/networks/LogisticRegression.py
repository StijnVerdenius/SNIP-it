import torch
import torch.nn as nn

from models.Pruneable import Pruneable

import numpy as np

class LogisticRegression(Pruneable):

    def __init__(self, device="cuda", hidden_dim=(10,), output_dim=2, input_dim=(1,), **kwargs):
        super(LogisticRegression, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        input_dim = int(np.prod(input_dim))

        self.layers = nn.Sequential(
            self.Linear(input_dim=input_dim, output_dim=output_dim),
        ).to(device)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)

if __name__ == '__main__':
    device = "cuda"

    mnist = torch.randn((21, 1, 28, 28)).to(device)
    cifar = torch.randn((21, 3, 32, 32)).to(device)
    imagenet = torch.randn((2, 4, 244, 244)).to(device)

    for test_batch in [mnist, cifar, imagenet]:
        conv = LogisticRegression(output_dim=10, input_dim=test_batch.shape[1:], device=device)

        print(conv.forward(test_batch).shape)