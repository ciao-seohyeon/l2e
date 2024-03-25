
# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License i∏NS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A set of tasks for quick iteration on meta-training."""
import functools
import jax
import gin
import haiku as hk
from learned_optimization.tasks import base
from learned_optimization.tasks import task_augmentation
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.fixed import conv
from learned_optimization.tasks.datasets import base as datasets_base
from tasks.conv import _ConvTask, _new_cross_entropy_pool_loss, _residual_cross_entropy_pool_loss
from tasks.transformer import _TransformerTask
from dataset.mnist import emnist_datasets, bloodmnist_datasets
from dataset.imagenet import imagenet32_datasets, tinyimagenet32_datasets

inner_bs = 128

def cifar_conv_task():
  base_model_fn = _new_cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
  datasets = image.cifar10_datasets(batch_size=inner_bs,
                                    image_size = (16, 16),
                                    convert_to_black_and_white=True)
  return conv._ConvTask(base_model_fn, datasets)

def imagenet_conv_task():
  base_model_fn = _new_cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
  datasets = tinyimagenet32_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def fashion_mnist_conv_task2():
  base_model_fn = _new_cross_entropy_pool_loss([32, 64, 64], jax.nn.relu, num_classes=10)
  datasets = image.fashion_mnist_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def fashion_mnist_task():
  datasets = image.fashion_mnist_datasets(batch_size=inner_bs, image_size=(8, 8))
  return image_mlp._MLPImageTask(datasets, [32])


def fashion_mnist_conv_task():
  base_model_fn = _new_cross_entropy_pool_loss([32], jax.nn.relu, num_classes=10)
  datasets = image.fashion_mnist_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def mnist_conv_task():
  base_model_fn = _new_cross_entropy_pool_loss([32], jax.nn.relu, num_classes=10)
  datasets = image.mnist_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def fashion_mnist_residual_conv_task():
  base_model_fn = _residual_cross_entropy_pool_loss([32], jax.nn.relu, num_classes=10)
  datasets = image.fashion_mnist_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def mnist_task():
  datasets = image.mnist_datasets(batch_size=inner_bs) 
  return image_mlp._MLPImageTask(datasets, [32]) 


def emnist_task():
  base_model_fn = _new_cross_entropy_pool_loss([32], jax.nn.relu, num_classes=10)
  datasets = emnist_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def bloodmnist_task():
  base_model_fn = _new_cross_entropy_pool_loss([32], jax.nn.relu, num_classes=8)
  datasets = bloodmnist_datasets(batch_size=inner_bs)
  return conv._ConvTask(base_model_fn, datasets)


def svhn_task():
  datasets = image.svhn_cropped_datasets(
      batch_size=inner_bs, image_size=(8, 8), convert_to_black_and_white=True)
  return image_mlp._MLPImageTask(datasets, [32]) 


def svhn_task2():
  datasets = image.svhn_cropped_datasets(
      batch_size=inner_bs)
  return image_mlp._MLPImageTask(datasets, [128, 64, 32]) 


def vgg_task():
    base_model_fn = _cross_entropy_pool_loss_vgg(activation_fn = jax.nn.relu, num_classes=10)
    datasets = image.cifar10_datasets(batch_size=128 , normalize_mean = (0.49,0.48,0.44), normalize_std = (0.2,0.2,0.2))
    return _ConvTask(base_model_fn, datasets, 5e-4)    
    
def language_task():
    cfg =  _cfg = {
      'num_heads': 5, # 8
      'd_model': 20,
      'num_layers': 1,
      'batch_size': 4,
      'sequence_length': 8, # 128
      'dropout_rate': 0.1
    }
    return _TransformerTask(cfg)
    
def task_to_augmented_task_family(task_fn):
  task_family = base.single_task_to_family(task=task_fn())
  return task_augmentation.ReparamWeightsFamily(task_family, "tensor", (0.01, 100))


def mnist_fmnist_cnn_depth_tasklist():
    task_family_list = []
    def mnist_fn():
      return image.mnist_datasets(batch_size=inner_bs)
    def fmnist_fn():
      return image.fashion_mnist_datasets(batch_size=inner_bs)
    def blood_fn():
      return bloodmnist_datasets(batch_size=inner_bs)
    def emnist_fn():
      return emnist_datasets(batch_size=inner_bs)
    
    data_mnist = datasets_base.LazyDataset(mnist_fn)
    data_fmnist = datasets_base.LazyDataset(fmnist_fn)
    data_bloodmnist = datasets_base.LazyDataset(blood_fn)
    data_emnist = datasets_base.LazyDataset(emnist_fn)

    num_label_mapping = [10, 10, 8, 10]
    activation = jax.nn.relu
    
    dataset_list = [data_mnist,data_fmnist,data_bloodmnist,data_emnist] 
    for layer_size in [4, 8, 16]: 
      for layer_num in [1, 2, 3, 4, 5]:
        for dataset, label_num in zip(dataset_list,num_label_mapping): 
          for residual in [True, False]:
            if residual:
              base_model_fn = _residual_cross_entropy_pool_loss([layer_size]*layer_num, activation, num_classes=label_num)
            else:
              base_model_fn = _new_cross_entropy_pool_loss([layer_size]*layer_num, activation, num_classes=label_num)
            task_ = conv._ConvTask(base_model_fn, dataset)
            task_family_list.append(base.single_task_to_family(task=task_))

    return task_family_list
