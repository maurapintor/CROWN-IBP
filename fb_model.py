
import copy
from collections import namedtuple

import torch

from bound_layers import BoundSequential
from config import load_config, config_modelloader
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



    for model, model_id, model_config in zip(models, model_names, config["models"]):
        # make a copy of global training config, and update per-model config
        eval_config = copy.deepcopy(global_eval_config)
        if "eval_params" in model_config:
            eval_config.update(model_config["eval_params"])
        model.eval()

        preprocessing = {'mean': 0.0, 'std': 1.0}
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1),
                                        preprocessing=preprocessing,
                                        device=device)

        return fmodel