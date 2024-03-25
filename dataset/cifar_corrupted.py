
import os
from typing import Tuple
from jax import core
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from learned_optimization.tasks.datasets import base
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple

Batch = Any


def cifar10_corrupted_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1,
                   intensity: int = 5) -> base.Datasets:
  splits = ("test")
  return base.preload_tfds_image_classification_datasets(
      f"cifar10_corrupted/gaussian_noise_{intensity}",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels)
  