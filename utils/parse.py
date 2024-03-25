import jax
from learned_optimization.tasks.datasets import image

from tasks.task import *
from tasks.resnet import _ResNetTask, _fc_resnet_loss_fn_frn, _fc_resnet_loss_fn_frn56
from tasks.conv import _ConvTask, _new_cross_entropy_pool_loss
from tasks.mlp import _MLPImageTask

from dataset.imagenet import tinyimagenet32_datasets, tinyimagenet64_datasets, imagenet32_datasets

from algorithms.l2e import L2E

import logging
import os 

def parse_lopt(lopt=None, config=None):
  if lopt == 'l2e':
    return L2E(config=config)
  else:
    raise NotImplementedError('Inappropriate meta learner name')

def parse_eval_task(task, batch_size , config=None):
  hidden_size = [40, 40]
  weight_decay = config.eval.wd 
  if task == 'mnist':
    datasets = image.mnist_datasets(batch_size=batch_size)
    task = _MLPImageTask(datasets=datasets, hidden_sizes=hidden_size, weight_decay=weight_decay) 
    return task
  elif task == 'fmnist':
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    task = _MLPImageTask(datasets=datasets, hidden_sizes=hidden_size, weight_decay=weight_decay) 
    return task
  elif task == 'fmnist_conv':
    datasets = image.fashion_mnist_datasets(batch_size=batch_size)
    base_model_fn = _new_cross_entropy_pool_loss([32], jax.nn.relu, num_classes=10)
    task = _ConvTask(base_model_fn, datasets, weight_decay)
    return task
  elif task == 'c10_frn':
    datasets = image.cifar10_datasets(batch_size=batch_size, normalize_mean=(0.49, 0.48, 0.44), normalize_std = (0.2, 0.2, 0.2))
    base_model_fn = _fc_resnet_loss_fn_frn(num_classes=datasets.extra_info['num_classes'])
    task = _ResNetTask(base_model_fn=base_model_fn, datasets=datasets, weight_decay=weight_decay)
    return task
  elif task == 'c100_frn':
    datasets = image.cifar100_datasets(batch_size=batch_size)
    base_model_fn = _fc_resnet_loss_fn_frn(num_classes=datasets.extra_info['num_classes'])
    task = _ResNetTask(base_model_fn=base_model_fn, datasets=datasets, weight_decay=weight_decay)
    return task
  elif task == 'tiny_frn':
    datasets = tinyimagenet64_datasets(batch_size=batch_size, image_size=(64, 64))
    base_model_fn = _fc_resnet_loss_fn_frn56(num_classes=datasets.extra_info['num_classes'])
    task = _ResNetTask(base_model_fn=base_model_fn, datasets=datasets, weight_decay=weight_decay)
    return task
  else:
    raise NotImplementedError('Inappropriate task name')
  

def write_json(obj, path: str, verbose: bool=False, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, **kwargs)
    if verbose:
        print(f"Results saved to {path}.")
        

def setup_logging(train_dir):
    logging.root.handlers = []
    logging_config = {
        'level': logging.INFO,
        'format': '[%(asctime)s %(filename)s: %(lineno)3d]: %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    logging_config['filename'] = os.path.join(train_dir, 'stdout.log')
    logging.basicConfig(**logging_config)