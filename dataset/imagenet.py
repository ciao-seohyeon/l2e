
import dataclasses
import numpy as np
import jax
from jax import core
import jax.numpy as jnp
import functools
import threading


import tensorflow as tf
import tensorflow_datasets as tfds

from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.datasets import base
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple

Batch = Any

def imagenet_image_classification_datasets(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: int,
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: int = 1000,
    shuffle_buffer_size: int = 10000,
    aug_flip_left_right: bool = False,
    aug_flip_up_down: bool = False,
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None,
    convert_to_black_and_white: Optional[bool] = False,
    cache: Optional[bool] = False,
) -> base.Datasets:
  num_classes_map = {
      "TinyImageNet200_x32": 200,
      "TinyImageNet200_x64": 200,
      "ImageNet1k_x32": 1000,
  }
  image_shapes_map = {
      "TinyImageNet200_x32": (32, 32, 3),
      "TinyImageNet200_x64": (64, 64, 3),
      "ImageNet1k_x32": (32, 32, 3)
  }
  if datasetname not in num_classes_map:
    raise ValueError(f"Trying to access an unsupported dataset: {datasetname}?")

  cfg = {
      "batch_size": batch_size,
      "image_size": image_size,
      "stack_channels": stack_channels,
      "prefetch_batches": prefetch_batches,
      "aug_flip_left_right": aug_flip_left_right,
      "aug_flip_up_down": aug_flip_up_down,
      "normalize_mean": normalize_mean,
      "normalize_std": normalize_std,
      "convert_to_black_and_white": convert_to_black_and_white,
  }

  def make_python_iter(split: str) -> Iterator[Batch]:
    features = {
        "image": tf.io.FixedLenFeature([], dtype=tf.string),
        "label": tf.io.FixedLenFeature([], dtype=tf.string)
    }

    if split == 'train':
      data = np.load(f'./dataset/{datasetname}/train_images.npy')
      label = np.load(f'./dataset/{datasetname}/train_labels.npy')
    else:
      data = np.load(f'./dataset/{datasetname}/test_images.npy')
      label = np.load(f'./dataset/{datasetname}/test_labels.npy')
    
    ds = tf.data.Dataset.from_tensor_slices({'image':data, 'label':label})
    ds = ds.repeat(-1)
    ds = ds.map(functools.partial(base._image_map_fn, cfg))
    
    if cache:
      ds = ds.cache()
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch_batches)
    return base.ThreadSafeIterator(base.LazyIterator(ds.as_numpy_iterator))

  if convert_to_black_and_white:
    shape = (batch_size,) + image_size + (1,)
  elif stack_channels == 1:
    shape = (batch_size,) + image_size + (image_shapes_map[datasetname][-1],)
  else:
    shape = (batch_size,) + image_size + (stack_channels,)

  abstract_batch = {
      "image": core.ShapedArray(shape, jnp.float32),
      "label": core.ShapedArray((batch_size,), jnp.int32),
  }

  return base.Datasets(
      *[make_python_iter(split) for split in splits],
      extra_info={"num_classes": num_classes_map[datasetname]},
      abstract_batch=abstract_batch)


def tinyimagenet32_datasets(batch_size: int,
                       image_size: Tuple[int, int] = (32, 32),
                       **kwargs) -> base.Datasets:
  splits = ("train", "inner_valid", "outer_valid", "test")
  return imagenet_image_classification_datasets(
      "TinyImageNet200_x32",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      cache=True,
      **kwargs)

def tinyimagenet64_datasets(batch_size: int,
                       image_size: Tuple[int, int] = (64, 64),
                       **kwargs) -> base.Datasets:
  splits = ("train", "inner_valid", "outer_valid", "test")
  return imagenet_image_classification_datasets(
      "TinyImageNet200_x64",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      cache=True,
      **kwargs)
  
  
def imagenet32_datasets(batch_size: int,
                      image_size: Tuple[int, int] = (32, 32),
                      **kwargs) -> base.Datasets:
  splits = ("train", "inner_valid", "outer_valid", "test")
  return imagenet_image_classification_datasets(
      "ImageNet1k_x32",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      cache=True,
      **kwargs)
