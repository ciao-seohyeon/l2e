from typing import Any, Optional, Sequence, Callable, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image


Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


def _new_cross_entropy_pool_loss(
    hidden_units: Sequence[int],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initializers: Optional[hk.initializers.Initializer] = None,
    norm_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    pool: str = "avg",
    num_classes: int = 10):
  """Haiku function for a conv net with pooling and cross entropy loss."""
  if not initializers:
    initializers = {}

  def _fn(batch):
    net = batch["image"]
    stride = 1
    for hs, ks, in zip(hidden_units, [3] * len(hidden_units)):
      net = hk.Conv2D(hs, ks, stride=stride)(net)
      net = activation_fn(net)
      net = norm_fn(net)

    net = hk.MaxPool(window_shape =2 , strides = 2, padding='VALID')(net)
    net = net.reshape(batch['image'].shape[0], -1)
    net = hk.Linear(20)(net) # hk.Linear(100)(net)
    logits = hk.Linear(num_classes)(net)

    labels = jax.nn.one_hot(batch["label"], num_classes)
    loss_vec = base.softmax_cross_entropy(labels=labels, logits=logits)

    return jnp.mean(loss_vec), logits

  return _fn


def _residual_cross_entropy_pool_loss(
    hidden_units: Sequence[int],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initializers: Optional[hk.initializers.Initializer] = None,
    norm_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    pool: str = "avg",
    num_classes: int = 10):
  """Haiku function for a conv net with pooling and cross entropy loss."""
  if not initializers:
    initializers = {}

  def _fn(batch):
    net = shortcut = batch["image"]
    stride = 1
    for i, (hs, ks) in enumerate(zip(hidden_units, [3] * len(hidden_units))):
      net = hk.Conv2D(hs, ks, stride=stride)(net)
      if i == len(hidden_units)-1:
        shortcut = hk.Conv2D(hs, 1, stride=1)(shortcut)
        net = activation_fn(net + shortcut)
      else:
        net = activation_fn(net)
      net = norm_fn(net)
    net = hk.MaxPool(window_shape =2 , strides = 2, padding='VALID')(net)

    net = net.reshape(batch['image'].shape[0], -1)
    logits = hk.Linear(num_classes)(net)

    labels = jax.nn.one_hot(batch["label"], num_classes)
    loss_vec = base.softmax_cross_entropy(labels=labels, logits=logits)

    return jnp.mean(loss_vec), logits

  return _fn

def _cross_entropy_pool_loss(
    hidden_units: Sequence[int],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initializers: Optional[hk.initializers.Initializer] = None,
    norm_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    pool: str = "avg",
    num_classes: int = 10):
  """Haiku function for a conv net with pooling and cross entropy loss."""
  if not initializers:
    initializers = {}

  def _fn(batch):
    net = batch["image"]
    strides = [2] + [1] * (len(hidden_units) - 1)
    for hs, ks, stride in zip(hidden_units, [3] * len(hidden_units), strides):
      net = hk.Conv2D(hs, ks, stride=stride)(net)
      net = activation_fn(net)
      net = norm_fn(net)

    if pool == "avg":
      net = jnp.mean(net, [1, 2])
    elif pool == "max":
      net = jnp.max(net, [1, 2])
    else:
      raise ValueError("pool type not supported")

    logits = hk.Linear(num_classes)(net)

    labels = jax.nn.one_hot(batch["label"], num_classes)
    loss_vec = base.softmax_cross_entropy(labels=labels, logits=logits)

    return jnp.mean(loss_vec), logits

  return _fn

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']



class _ConvTask(base.Task):
  """Helper class to construct tasks with different configs."""

  def __init__(self, base_model_fn, datasets, weight_decay, with_state=False):
    super().__init__()
    self._mod = hk.transform_with_state(base_model_fn)
    self.datasets = datasets
    self._with_state = with_state
    self.weight_decay = weight_decay

  def init(self, key) -> Params:
    params, unused_state = self.init_with_state(key)
    return params

  def init_with_state(self, key) -> Tuple[Params, ModelState]:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch)

  def loss(self, params, key, data):
    l2_norm = 0.5 * self.weight_decay * sum([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)])
    loss, _, _ = self.loss_with_state(params, None, key, data)
    return loss + l2_norm

  def logit(self, params, key, data):
    l2_norm = 0.5 * self.weight_decay * sum([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)])
    loss, logits, _ = self.loss_with_state(params, None, key, data)
    return logits
  
  def nll(self, params, key, data):
    nll1, _, _ = self.loss_with_state(params, None, key, data)
    # l2_norm = 0.5 * self.weight_decay * sum([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)])
    return nll1
  
  def loss_with_state(self, params, state, key, data):
    loss, logits, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, logits, state

  def loss_with_state_and_aux(self, params, state, key, data):
    (loss, logits), state = self._mod.apply(params, state, key, data)
    # loss, state = self._mod.apply(params, state, key, data)

    return loss, logits, state, {}

  def normalizer(self, loss):
    return jnp.clip(loss, 0,
                    1.5 * jnp.log(self.datasets.extra_info["num_classes"]))