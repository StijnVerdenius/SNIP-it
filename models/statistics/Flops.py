from typing import List

import torch
from torch.utils.hooks import RemovableHandle

from models.networks.assisting_layers.L0_Layers import L0Linear, L0Conv2d


class FLOPCounter:

    """
    Estimates current FLOPS usage of one sample or cumulatively
    Loosely based on implementation from:
    Rethinking the Value of Network Pruning
    https://arxiv.org/abs/1810.05270
    Sourced from their github at:
    https://github.com/Eric-mingjie/rethinking-network-pruning
    """

    def __init__(self, model, test_batch, batch_size, device="cuda"):
        self.batch_size = batch_size
        self.device = device
        self.test_batch = test_batch.to(self.device)
        self.model = model
        self.cumulative_flops_counter = 0
        self.last_batches_seen = 0

    def count_flops(self, batches_seen):

        difference_batches_seen = batches_seen - self.last_batches_seen

        self.last_batches_seen = batches_seen

        counter = [torch.zeros([1]).to(self.device)]

        def conv_hook(self, input, output):
            input_channels, input_height, input_width = input[0][0].shape
            output_channels, output_height, output_width = output[0].shape

            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * input_channels
            bias_ops = 1 if self.bias is not None else 0
            kernel_output_ops = output_channels * (kernel_ops + bias_ops)
            flops = kernel_output_ops * output_height * output_width
            counter[0] += flops

        def linear_hook(self, input, output):
            weight_ops = self.weight.nelement()

            bias_ops = 1 if self.bias is not None else 0
            flops = weight_ops + bias_ops * output.shape[-1]
            counter[0] += flops

        def conv_hook_l0(self, input, output):
            input_channels, input_height, input_width = (input[0][0].sum(dim=(1, 2)) != 0).sum().item(), *input[0][
                                                                                                              0].shape[
                                                                                                          1:]
            output_channels, output_height, output_width = (output[0].sum(dim=(1, 2)) != 0).sum().item(), *output[
                                                                                                               0].shape[
                                                                                                           1:]

            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * input_channels
            bias_ops = 1 if self.bias is not None else 0
            kernel_output_ops = output_channels * (kernel_ops + bias_ops)
            flops = kernel_output_ops * output_height * output_width
            counter[0] += flops

        def linear_hook_l0(self, input, output):
            weight_ops = (self.sample_weights() != 0).sum()

            bias_ops = 1 if self.bias is not None else 0
            flops = weight_ops + bias_ops * (output[0] != 0).sum()
            counter[0] += flops

        def bn_hook(self, input, output):
            flops = input[0].nelement() * 2
            counter[0] += flops

        def bn_hook_l0(self, input, output):
            flops = (input[0] != 0).sum() * 2
            counter[0] += flops

        def relu_hook(self, input, output):
            flops = input[0].nelement()
            counter[0] += flops

        def relu_hook_l0(self, input, output):
            flops = (input[0] != 0).sum()
            counter[0] += flops

        def pooling_hook(self, input, output):
            output_channels, output_height, output_width = output[0].shape
            kernel_ops = self.kernel_size * self.kernel_size
            flops = kernel_ops * output_channels * output_height * output_width
            counter[0] += flops

        def apply_hooks(model) -> List[RemovableHandle]:
            handles = []
            for module in model.modules():
                children = list(module.children())
                if not children:
                    if self.model.l0:
                        if isinstance(module, L0Conv2d):
                            handles.append(module.register_forward_hook(conv_hook_l0))
                        if isinstance(module, L0Linear):
                            handles.append(module.register_forward_hook(linear_hook_l0))
                        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                            handles.append(module.register_forward_hook(bn_hook_l0))
                        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
                            handles.append(module.register_forward_hook(relu_hook_l0))
                    else:
                        if isinstance(module, torch.nn.Conv2d):
                            handles.append(module.register_forward_hook(conv_hook))
                        if isinstance(module, torch.nn.Linear):
                            handles.append(module.register_forward_hook(linear_hook))
                        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                            handles.append(module.register_forward_hook(bn_hook))
                        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
                            handles.append(module.register_forward_hook(relu_hook))
                    if isinstance(module, torch.nn.MaxPool2d) or isinstance(module, torch.nn.AvgPool2d):
                        handles.append(module.register_forward_hook(pooling_hook))
            return handles

        handles = apply_hooks(self.model)
        self.model.forward(self.test_batch)
        flops_per_sample = counter[0].item()
        self.cumulative_flops_counter += flops_per_sample * difference_batches_seen * self.batch_size
        for handle in handles:
            handle.remove()

        return flops_per_sample, self.cumulative_flops_counter
