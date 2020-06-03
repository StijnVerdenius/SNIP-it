import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from utils.constants import NUM_WORKERS, FLIP_CHANCE, DATASET_PATH, IMAGENETTE_DIR, TINY_IMAGNET_DIR, \
    IMAGEWOOF_DIR

"""
Handles loading datasets
"""

def get_imagenette_loaders(arguments):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transformers = transforms.Compose(
        (
            [] if arguments.preload_all_data
            else [
                transforms.RandomHorizontalFlip(p=FLIP_CHANCE),
            ]
        ) +
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    train_set = torchvision.datasets.ImageFolder(os.path.join(IMAGENETTE_DIR, "train"), transform=transformers)
    test_set = torchvision.datasets.ImageFolder(os.path.join(IMAGENETTE_DIR, "val"), transform=transformers)

    return load(arguments, test_set, train_set)


def get_imagewoof_loaders(arguments):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transformers = transforms.Compose(
        (
            [] if arguments.preload_all_data
            else [
                transforms.RandomHorizontalFlip(p=FLIP_CHANCE),
            ]
        ) +
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    train_set = torchvision.datasets.ImageFolder(os.path.join(IMAGEWOOF_DIR, "train"), transform=transformers)
    test_set = torchvision.datasets.ImageFolder(os.path.join(IMAGEWOOF_DIR, "val"), transform=transformers)

    return load(arguments, test_set, train_set)


def get_mnist_loaders(arguments):
    transformers = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(
        DATASET_PATH,
        train=True,
        download=True,
        transform=transformers
    )
    test_set = datasets.MNIST(
        DATASET_PATH,
        train=False,
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set)


def get_cifar10_loaders(arguments):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    train_transforms = transforms.Compose(
        (
            [] if arguments.preload_all_data
            else [
                transforms.RandomHorizontalFlip(p=FLIP_CHANCE),
            ]
        ) +
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )
    train_set = datasets.CIFAR10(DATASET_PATH, train=True, download=True, transform=train_transforms)
    test_set = datasets.CIFAR10(DATASET_PATH, train=False, download=True, transform=test_transforms)
    return load(arguments, test_set, train_set)


def preloading(arguments, test_set, train_set):
    print("preloading data")
    train_images, train_labels = zip(*train_set)
    test_images, test_labels = zip(*test_set)
    train_images = torch.stack(train_images, dim=0).to(arguments.device)
    train_labels = torch.tensor(train_labels).to(arguments.device)
    test_images = torch.stack(test_images, dim=0).to(arguments.device)
    test_labels = torch.tensor(test_labels).to(arguments.device)
    # noinspection PyTypeChecker
    train_loader, test_loader = PersonalDataLoader(train_images, train_labels, arguments.batch_size,
                                                   horizontal_flips=True, device=arguments.device), \
                                PersonalDataLoader(test_images, test_labels, arguments.batch_size,
                                                   device=arguments.device)
    return test_loader, train_loader


def load(arguments, test_set, train_set):
    if arguments.tuning:
        print("Running in tuning mode, omit testset")
        total_length = len(train_set)
        train_length = int(0.8 * total_length)
        val_length = total_length - train_length
        train_set, test_set = torch.utils.data.random_split(train_set, [train_length, val_length])

    if arguments.random_shuffle_labels:
        print("randomly shuffling labels")
        test_set.targets = test_set.targets[torch.randperm(len(test_set.targets))]
        train_set.targets = train_set.targets[torch.randperm(len(train_set.targets))]

    if arguments.preload_all_data:
        test_loader, train_loader = preloading(arguments, test_set, train_set)

    else:

        test_loader, train_loader = traditional_loading(arguments, test_set, train_set)
    return train_loader, test_loader


def get_cifar100_loaders(arguments):
    if arguments.preload_all_data: raise NotImplementedError

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DATASET_PATH, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(p=0.2),
                              transforms.RandomAffine(5),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean,
                                                   std=std)
                          ])),
        batch_size=arguments.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DATASET_PATH, train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=mean,
                                                                             std=std)

                                                        ])),
        batch_size=arguments.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    return train_loader, test_loader


def get_omniglot_loaders(arguments):
    if arguments.preload_all_data: raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        datasets.Omniglot(DATASET_PATH, background=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomAffine(10),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
        batch_size=arguments.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.Omniglot(DATASET_PATH, background=False, download=True,
                          transform=transforms.Compose([  # transforms.RandomCrop(70),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
        batch_size=arguments.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    return train_loader, test_loader


def get_imagenet_loaders(arguments):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transformers = transforms.Compose(
        (
            [] if arguments.preload_all_data
            else [
                transforms.RandomHorizontalFlip(p=FLIP_CHANCE),
            ]
        ) +
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    train_set = torchvision.datasets.ImageFolder(os.path.join(TINY_IMAGNET_DIR, "train"), transform=transformers)
    test_set = torchvision.datasets.ImageFolder(os.path.join(TINY_IMAGNET_DIR, "val"), transform=transformers)

    return load(arguments, test_set, train_set)


def traditional_loading(arguments, test_set, train_set):
    workers = NUM_WORKERS
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=arguments.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=arguments.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=workers
    )
    return test_loader, train_loader


def get_rubbish_loaders(arguments=None):
    bs = 10 if arguments is None else arguments.batch_size
    train_loader = torch.utils.data.DataLoader(
        RubbishSet(),
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS

    )

    test_loader = torch.utils.data.DataLoader(
        RubbishSet(),
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS

    )

    return train_loader, test_loader


#### Classes

class RubbishSet(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        class_ = random.choice([0, 1])
        tensor = np.random.normal(class_, 0.2, (3, 3))
        return tensor, class_

    def __len__(self):
        return 10000


class PersonalDataLoader:

    # def _reset(self):
    def __init__(self, data, labels, batch_size: int, device="cuda", horizontal_flips=False):
        self.horizontal_flips = horizontal_flips
        self.labels = labels
        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.length = torch.LongTensor([(len(data) // batch_size) + 1]).to(device)
        self.length_orgi = len(data)
        self.indices = None
        self.flips = None
        self.counter = torch.zeros([1], device=self.device).long()
        self.one = torch.ones([1], device=self.device).long()

    def __len__(self):
        return self.length

    def __iter__(self):
        self.indices = (
                torch.randperm(self.length.item() * self.batch_size, device=self.device) % self.length_orgi).view(
            -1, self.batch_size)
        if self.horizontal_flips:
            self.flips = torch.bernoulli(torch.empty(self.data.shape[0]).to(self.device), p=FLIP_CHANCE).bool()
            self.data[self.flips] = self.data[self.flips].flip(-1)
        self.counter -= self.counter
        return self

    def __next__(self):
        if self.counter >= self.length:
            if self.horizontal_flips:
                self.data[self.flips] = self.data[self.flips].flip(-1)
            raise StopIteration
        else:
            batch = self.__getitem__(self.counter)
            self.counter += self.one
            return batch

    def __getitem__(self, item):
        return self.data[
                   self.indices[item].squeeze()
               ], \
               self.labels[
                   self.indices[item].squeeze()
               ]
