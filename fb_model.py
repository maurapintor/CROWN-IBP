
import copy
from collections import namedtuple
from functools import partial

import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms

from bound_layers import BoundSequential
from eps_scheduler import EpsilonScheduler
from config import load_config, get_path, config_modelloader, config_dataloader
from argparser import argparser
from train import Train, Logger
import foolbox as fb

class BoundSequential2(BoundSequential):
    def __call__(self, *args, **kwargs):
        kwargs['method_opt'] = 'forward'
        return super(BoundSequential2, self).__call__(*args, **kwargs)

def create():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args_tuple = namedtuple('args_tuple', "config overrides_dict model_subset path_prefix")
    args = args_tuple(config='config/mnist_dm-large_0.4.json', overrides_dict={},
                      model_subset={}, path_prefix='models_crown-ibp_dm-large')
    config = load_config(args)

    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain=True,
                                             cuda=torch.cuda.is_available())

    model, model_id, model_config = models[0], model_names[0], config["models"][0]

    eval_config = copy.deepcopy(global_eval_config)
    if "eval_params" in model_config:
        eval_config.update(model_config["eval_params"])
    model = BoundSequential.convert(model, eval_config["method_params"]["bound_opts"])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    preprocessing = {'mean': 0.0, 'std': 1.0}
    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1),
                                    preprocessing=preprocessing,
                                    device=device)

    return fmodel
