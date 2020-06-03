
# SNIP-it / SNAP-it
## (Un)structured Pruning via Iterative Ranking of Sensitivity Statistics

[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB.svg?logo=python)](https://www.python.org/) [![PyTorch 1.4](https://img.shields.io/badge/PyTorch-1.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.4.0/) [![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

This repository is the official implementation of the paper [Pruning via Iterative Ranking of Sensitivity Statistics](https://arxiv.org/abs/2006.00896). 
Currently under review. Please use this preliminary BibTex entry when reffering to our work:

```reference
@article{verdenius2020pruning,
       author = {{Verdenius}, Stijn and {Stol}, Maarten and {Forr{\'e}}, Patrick},
        title = "{Pruning via Iterative Ranking of Sensitivity Statistics}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
         year = 2020,
        month = jun,
          eid = {arXiv:2006.00896},
        pages = {arXiv:2006.00896},
archivePrefix = {arXiv},
       eprint = {2006.00896},
 primaryClass = {cs.LG},
}
```

### Content

The repository implements novel pruning / compression algorithms for deep learning / neural networks. Aditionally, it implements the shrinkage of actual tensors to really benefit from structured pruning without external hardware libraries. We implement:

- Structured (node) pruning before training
- Structured (node) pruning during training
- Unstructured (weight) pruning before training 
- Unstructured (weight) pruning during training

### Setup

- Install virtualenv

> `pip3 install virtualenv`

- Create environment

> `virtualenv -p python3 ~/virtualenvs/SNIPIT`

- Activate environment

> `source ~/virtualenvs/SNIPIT/bin/activate`

- Install requirements:

> `pip install -r requirements.txt`

- If you mean to run the Imagenette dataset: download that from [its github repository](https://github.com/fastai/imagenette) and unpack in `/gitignored/data/`, then replace CIFAR10 with IMAGENETTE below to run. Additional datasets can be added in a similar way (Imagewoof, tiny-imagenet, etc.)

### Training Examples & Results

Some examples of training the models from the paper.

#### Structured Pruning (SNAP-it)

To run training for SNAP-it - our structured pruning before training algorithm - with a VGG16 on CIFAR10, run the following:

```train
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion SNAPit --pruning_limit 0.93 --epochs 80
```

<img src="./pictures/__structured-VGG16-CIFAR10_acc_node_sparse.png" alt="drawing" width="500"/>

| accuracy-drop | weight sparsity | node sparsity | cumulative training FLOPS reduction |
|---------------|-----------------|---------------|-------------------------------------|
| -1%           | 99%             | 93%           | 8 times                             |

#### Unstructured Pruning (SNIP-it)

To run training for SNIP-it - our unstructured pruning algorithm - with a ResNet18 on CIFAR10, run one of the following:

```train
## during training
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion SNIPitDuring --pruning_limit 0.98 --outer_layer_pruning --epochs 80 --prune_delay 4 --prune_freq 4
## before training
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion SNIPit --pruning_limit 0.98 --outer_layer_pruning --epochs 80 
```
<img src="./pictures/__unstructured-ResNet18-CIFAR10_acc_weight_sparse.png" alt="drawing" width="400"/>

|                  | accuracy-drop | weight sparsity |
|------------------|---------------|-----------------|
| SNIP-it (during) | -0%           | 98%             |
| SNIP-it (before) | -4%           | 98%             |

#### Adversarial Evaluation

To evaluate a model on adversarial attacks (for now only supported on unstructured), run:

```eval
python main.py --eval --model MLP5 --data_set MNIST --checkpoint_name <see_results_folder> --checkpoint_model MLP5_finished --attack CarliniWagner
```

### Arguments

The regulare arguments for running are the following. Aditionally, there are some more found in utils/config_utils.py.

| **argument**          | **description**                                            |**type**|
|-----------------------|------------------------------------------------------------|-------|
| --model               | The neural network architecture from `/models/networks/`   | str   |
| --data_set            | The dataset from `/utils/dataloaders.py`                   | str   |
| --prune_criterion     | The pruning criterion from `/models/criterions/`           | str   |
| --batch_size          | The batch size                                             | int   |
| --optimizer           | The optimizer model class from [ADAM, SGD & RMSPROP]       |  str  |
| --loss                | The loss function from `/models/losses/`                   | str   |
| --train_scheme        | The training scheme from `/models/trainers/` (if applicable)| str   |
| --test_scheme         | The testing scheme from `/models/testers/` (if applicable) | str   |
| --eval                | Add to run in test mode                                    | bool  |
| --attack              | Name of adersarial attack if that is the test_scheme       | str   |
| --device              | Device [cuda or cpu]                                       | srt   |
| --run_name            | Extra run identification for generated run folder          | str   |
| --checkpoint_name     | Load from this checkpoint folder if not None               | str   |
| --checkpoint_model    | Load this model from checkpoint_name                       | str   |
| --outer_layer_pruning | Prunes outer layers too. Use iff pruning unstructured      | bool  |
| --enable_rewinding    | Does rewinding of weights (for IMP)                        | bool  |
| --rewind_epoch        | Epoch to rewind to                                         | int   |
| --l0                  | Run with L0-reg layers, overrides some other options       | bool  |
| --l0_reg              | L0 regularisation hyperparameter                           | float |
| --hoyer_square        | Run with hoyersquare, overrides some other options         | bool  |
| --group_hoyer_square  | Run with grouphoyersquare, overrides some other options    | bool  |
| --hoyer_reg           | Hoyer regularisation hyperparameter                        | float |
| --learning_rate       | Learning rate                                              | float |
| --pruning_limit       | Final sparsity endeavour for applicable pruning criterions | float |
| --pruning_rate        | Outdated pruning_limit, still used for UnstructuredRandom  | float |
| --snip_steps          | `S` from paper algorithm box 1. Number of pruning steps      | int   |
| --epochs              | How long to train for                                      | int   |
| --prune_delay         | `Tau` from paper algorithm box 1. How long to start pruning  | int   |
| --prune_freq          | `Tau` from paper algorithm box 1 again. How often to prune   | int   |
| --seed                | Random seed to run with                                    | int   |
| --tuning              | Run with train and held out validationset, omit testset    | bool  |

Some notes:

> - please note that as of now, residual connections (e.g. ResNets) and structured pruning are not supported together.
> - please note that as of now, structured pruning and `--outer_layer_pruning` are not supported together.
> - please note that as of now, if running unstructured pruning, you should also run with `--outer_layer_pruning`.


### Codebase Design

- Codebase is built modularly so that every criterion or model that is added to its designated folder, provided its filename is equal to its classname, can be ran via string argument immediately. This way its easily extendable.
- The same goes for training schemes; implemented here as classes and automatically loaded in by string reference. When you need new functionality concerning one aspect of training you can simply inheret the `DefaultTrainer` and then override only that function you need differently. Alternativly, you can make your own training scheme, the sky is the limit!
- All entrypoints go through `main.py`, where the required models are loaded and thereafter redirected to the right training or testing scheme.
- All results show up at the path `/gitignored/results/` in its own (dated) folder. In here you find a copy of the codebase at the time of execution, its calling command, tensorboard output, saved models and logs.
- In the file utils/autoconfig.json certain automatic configurations get set to make it easier to run different models in sequence. You can disable this with `--disable_autoconfig`, but it is *strongly recommended against*.


### How to run the other baselines

```
## unpruned baselines
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion EmptyCrit --epochs 80 --pruning_limit 0.0
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion EmptyCrit --epochs 80 --pruning_limit 0.0

## structured baselines
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion StructuredRandom --pruning_limit 0.93 --epochs 80
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion GateDecorators --pruning_limit 0.93 --epochs 70 --checkpoint_name <unpruned_results_folder> --checkpoint_model VGG16_finished 
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion EfficientConvNets --pruning_limit 0.93 --epochs 80 --prune_delay 69 --prune_freq 1
python3 main.py --model VGG16 --data_set CIFAR10 --prune_criterion GroupHoyerSquare --hoyer_reg <REG> --epochs 80 --prune_delay 69 --prune_freq 1 --group_hoyer_square
python3 main.py --model VGG16 --data_set CIFAR10 --l0_reg <REG> --epochs 160 --l0

## unstructured baselines
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion UnstructuredRandom --pruning_rate 0.98 --pruning_limit 0.98 --outer_layer_pruning --epochs 80
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion <SNIP or GRASP> --pruning_limit 0.98 --outer_layer_pruning --epochs 80
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion HoyerSquare --hoyer_reg <REG> --outer_layer_pruning --epochs 80 --prune_delay 69 --prune_freq 1 --hoyer_square
python3 main.py --model ResNet18 --data_set CIFAR10 --prune_criterion IMP --pruning_limit 0.98 --outer_layer_pruning --epochs 80 --prune_delay 4 --prune_freq 4 --enable_rewinding --rewind_to 6
```
### Licence

[Licence](LICENSE) 
