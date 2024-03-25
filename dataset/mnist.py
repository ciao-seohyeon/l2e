import os
from typing import Tuple
from jax import core
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from learned_optimization.tasks.datasets import base
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple

Batch = Any


def emnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return base.preload_tfds_image_classification_datasets(
      "emnist",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels)
  

def kmnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return base.preload_tfds_image_classification_datasets(
      "kmnist",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels)


def medmnist_image_classification_datasets(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: int,
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: int = 1000,
    shuffle_buffer_size: int = 10000,
    convert_to_black_and_white: Optional[bool] = False,
    cache: Optional[bool] = False,
):   

    num_classes_map = {
        "pathmnist": 9,
        "bloodmnist": 8,
        "dermamnist": 7
    }
    image_shapes_map = {
        "pathmnist": (28, 28, 3),
        "bloodmnist": (28, 28, 3),
        "dermamnist": (28, 28, 3),
    }
    if datasetname not in num_classes_map:
        raise ValueError(f"Trying to access an unsupported dataset: {datasetname}?")

    def make_python_iter(split: str) -> Iterator[Batch]:
        npz_file = np.load(os.path.join("./dataset/medmnist", "{}.npz".format(datasetname)))

        if split == 'train':
            data = npz_file['train_images']
            label = npz_file['train_labels']
        elif split == 'inner_valid' or split == 'outer_valid':
            data = npz_file['val_images']
            label = npz_file['val_labels']
        elif split == 'test':
            data = npz_file['test_images']
            label = npz_file['test_labels']
        else:
            raise ValueError

        ds = tf.data.Dataset.from_tensor_slices({'image':data, 'label':label.reshape(-1,)})
        
        def preprocess(r):
            r['image'] = tf.cast(r['image'], tf.float32) / 255.0
            return r
        
        ds = ds.repeat(-1)
        ds = ds.map(preprocess)
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


def pathmnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1) -> base.Datasets:
    splits = ("train", "inner_valid", "outer_valid", "test")
    return medmnist_image_classification_datasets(
      "pathmnist",
      splits,
      batch_size=batch_size,
      image_size=image_size)

def dermamnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1) -> base.Datasets:
    splits = ("train", "inner_valid", "outer_valid", "test")
    return medmnist_image_classification_datasets(
      "dermamnist",
      splits,
      batch_size=batch_size,
      image_size=image_size)

def bloodmnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1) -> base.Datasets:
    splits = ("train", "inner_valid", "outer_valid", "test")
    return medmnist_image_classification_datasets(
      "bloodmnist",
      splits,
      batch_size=batch_size,
      image_size=image_size)