from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models.criterions.General import General


class StructuredRandom(General):

    """
    Original creation from our paper: todo
    Implements Random (structured before training), a surprisingly strong baseline.
    Additionally, this class contains most of the code the actually reduce pytorch tensors, in order to obtain speedup
    See SNAP.py for comments for functionality of functions here
    """

    def __init__(self, *args, **kwargs):
        super(StructuredRandom, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, **kwargs):

        in_indices, out_indices = None, None

        modules = [(name, elem)
                   for name, elem in self.model.named_modules()
                   if isinstance(elem, (torch.nn.Linear, torch.nn.Conv2d))]
        last_is_conv = False
        for i, (name, module) in enumerate(modules):
            num_params = np.prod(module.weight.shape)
            in_indices, last_is_conv, now_is_conv, out_indices = self.get_inclusion_vectors(i,
                                                                                            in_indices,
                                                                                            last_is_conv,
                                                                                            module,
                                                                                            modules,
                                                                                            out_indices,
                                                                                            percentage)

            self.handle_input(in_indices, module, now_is_conv)
            self.handle_output(out_indices, now_is_conv, module, name)
            params_left = np.prod(module.weight.shape)
            pruned = num_params - params_left
            print("pruning", pruned, "percentage", (pruned) / num_params, "length_nonzero", num_params)

        self.model.mask = {name + ".weight": torch.ones_like(module.weight.data).to(self.device)
                           for name, module in self.model.named_modules()
                           if isinstance(module, (nn.Linear, nn.Conv2d))
                           }

        print(self.model)
        print("Final percentage: ", self.model.pruned_percentage)

    def handle_output(self, indices, is_conv, module, name):
        weight = module.weight
        module.update_output_dim(indices.sum())
        self.handle_batch_norm(indices, indices.sum(), name)
        if is_conv:
            weight.data = weight[indices, :, :, :]
            weight.grad.data = weight.grad.data[indices, :, :, :]
        else:
            weight.data = weight[indices, :]
            weight.grad.data = weight.grad.data[indices, :]
        self.handle_bias(indices, module)

    def handle_bias(self, indices, module):
        bias = module.bias
        bias.data = bias[indices]
        bias.grad.data = bias.grad.data[indices]

    def handle_batch_norm(self, indices, n_remaining, name):
        batchnorm = [val for key, val in self.model.named_modules() if
                     key == name[:-1] + str(int(name[-1]) + 1)]
        if len(batchnorm) == 1:
            batchnorm = batchnorm[0]
        else:
            return
        if isinstance(batchnorm, (nn.BatchNorm2d, nn.BatchNorm1d)):
            batchnorm.num_features = n_remaining
            from_size = len(batchnorm.bias.data)
            batchnorm.bias.data = batchnorm.bias[indices]
            batchnorm.bias.grad.data = batchnorm.bias.grad[indices]
            batchnorm.weight.data = batchnorm.weight[indices]
            batchnorm.weight.grad.data = batchnorm.weight.grad[indices]
            for buffer in batchnorm.buffers():
                if buffer.data.shape == indices.shape:
                    buffer.data = buffer.data[indices]
            print(f"trimming {name} nodes from {from_size} to {len(batchnorm.bias.data)}")

    def handle_input(self, indices, module, now_is_conv):
        module.update_input_dim(indices.sum())
        if now_is_conv:
            module.weight.data = module.weight.data[:, indices, :, :]
            module.weight.grad.data = module.weight.grad.data[:, indices, :, :]
        else:
            module.weight.data = module.weight.data[:, indices]
            module.weight.grad.data = module.weight.grad.data[:, indices]

    def get_inclusion_vectors(self, i, in_indices, last_is_conv, module, modules, out_indices, percentage):
        param = module.weight
        dims = param.shape[:2]  # out, in
        if in_indices is None:
            in_indices = torch.ones(dims[1])
            if self.model._outer_layer_pruning:
                in_indices = self.get_in_vector(dims, percentage)
        else:
            in_indices = out_indices
        out_indices = self.get_out_vector(dims, percentage)
        is_last = (len(modules) - 1) == i
        if is_last and not self.model._outer_layer_pruning:
            out_indices = torch.ones(dims[0])
        now_is_fc = isinstance(module, torch.nn.Linear)
        now_is_conv = isinstance(module, torch.nn.Conv2d)
        if last_is_conv and now_is_fc:
            ratio = param.shape[1] // in_indices.shape[0]
            in_indices = torch.repeat_interleave(in_indices, ratio)
        last_is_conv = now_is_conv
        if in_indices.sum() == 0:
            in_indices[randint(0, len(in_indices) - 1)] = 1
        if out_indices.sum() == 0:
            out_indices[randint(0, len(out_indices) - 1)] = 1
        return in_indices.bool(), last_is_conv, now_is_conv, out_indices.bool()

    def get_out_vector(self, dims, percentage):
        return (init.sparse(torch.empty(dims), percentage)[:, 0] != 0).long()

    def get_in_vector(self, dims, percentage):
        return (init.sparse(torch.empty(dims), percentage)[0, :] != 0).long()
