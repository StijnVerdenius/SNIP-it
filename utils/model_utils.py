import importlib
import inspect
from typing import List

import torch.nn as nn
import torch.optim as opt

from utils.constants import *
from utils.data_loaders import *

models = {x: {} for x in types}

"""
Handles loading/init/saving of models and more
"""

def _read_all_class_names():
    """
    private function that imports all class references in a dictionary
    """

    for typ in types:
        for name in os.listdir(os.path.join(".", MODELS_DIR, typ)):
            if not ("__" in name) and ".py" in name:
                short_name = str(name.split(".")[0])
                short_name: str
                module = importlib.import_module(f"models.{typ}.{short_name}")
                class_reference = getattr(module, short_name)
                models[typ][short_name] = class_reference

    # hardcoded ones
    models[OPTIMS] = {}
    models[OPTIMS]["ADAM"] = opt.Adam
    models[OPTIMS]["RMSPROP"] = opt.RMSprop
    models[OPTIMS]["SGD"] = opt.SGD

    models[DATASETS] = {}
    models[DATASETS]["RUBBISH"] = get_rubbish_loaders
    models[DATASETS]["CIFAR10"] = get_cifar10_loaders
    models[DATASETS]["CIFAR100"] = get_cifar100_loaders
    models[DATASETS]["TINYIMAGENET"] = get_imagenet_loaders
    models[DATASETS]["IMAGENETTE"] = get_imagenette_loaders
    models[DATASETS]["IMAGEWOOF"] = get_imagewoof_loaders
    models[DATASETS]["MNIST"] = get_mnist_loaders
    models[DATASETS]["OMNIGLOT"] = get_omniglot_loaders


def find_right_model(type: str, name: str, **kwargs):
    """
    returns model with arguments given a string name-tag
    """

    return models[type][name](**kwargs)


def detach_list(items):
    for i, x in enumerate(items):
        items[i] = x.detach()


def delete_list(items):
    for i, x in enumerate(items):
        del items[i], x


def save_models(models: List[nn.Module],
                suffix: str):
    """
    Saves current state of models
    """

    for mod in models:
        modstr = str(mod.__class__).split(".")[-1].replace("'>", "_")
        state = mod.state_dict()

        DATA_MANAGER.save_python_obj(state,
                                     os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, MODELS_DIR, modstr + suffix),
                                     print_success=False)


def calculate_accuracy(targets, output, *ignored):
    output = torch.nn.functional.softmax(output, dim=-1).detach()
    _, classifications = output.detach().max(dim=-1)
    return (targets.eq(classifications)).float().mean()


def assert_type(content, expected_type):
    """ makes sure type is respected"""

    assert_non_empty(content)
    func = inspect.stack()[1][3]
    assert isinstance(content, type(expected_type)), "No {} entered in {} but instead value {}".format(
        str(expected_type),
        func,
        str(content))


def assert_non_empty(content):
    """ makes sure not None or len()==0 """

    func = inspect.stack()[1][3]
    assert not (content is None), "Content is null in {}".format(func)
    if type(content) is list or type(content) == str:
        assert len(content) > 0, "Empty {} in {}".format(type(content), func)


# needed to load in class references
_read_all_class_names()
