from typing import Any, Mapping, Tuple, Sequence
import functools
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks.datasets import image
import numpy as onp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
from learned_optimization.time_filter import model_paths
from learned_optimization.time_filter import time_model

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


class _MLPImageTask(base.Task):
  """MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               weight_decay,
               act_fn=jax.nn.relu,
               dropout_rate=0.0):
    super().__init__()
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.datasets = datasets
    self.weight_decay = weight_decay

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return hk.nets.MLP(
          sizes, activation=act_fn)(
              inp, dropout_rate=dropout_rate, rng=hk.next_rng_key())

    self._mod = hk.transform(_forward)

  def init(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    l2_norm = 0.5 * self.weight_decay * sum([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)])
  
    return jnp.mean(vec_loss) + l2_norm  # ,  jnp.mean(vec_loss) # (jnp.mean(vec_loss),l2_norm)

  def nll(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray: # should debug
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)
  
  def logit(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    return self._mod.apply(params, key, data["image"])
  
  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricImageMLP(base.TaskFamily):
  """A parametric image model based on an MLP."""

  def __init__(self,
               datasets: datasets_base.Datasets,
               num_classes: int,
               hidden_layers: Sequence[int] = (32, 32)):

    super().__init__()
    self.hidden_layers = hidden_layers
    self.datasets = datasets
    self.num_clases = num_classes

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    act_key, init_key = jax.random.split(key, 2)
    return cfgobject.CFGNamed(
        "ParametricImageMLP", {
            "initializer": parametric_utils.SampleInitializer.sample(init_key),
            "activation": parametric_utils.SampleActivation.sample(act_key),
        })

  def task_fn(self, task_params) -> base.Task:

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      act_fn = parametric_utils.SampleActivation.get_dynamic(
          task_params.values["activation"])
      return hk.nets.MLP(
          tuple(self.hidden_layers) + (self.num_clases,),
          w_init=parametric_utils.SampleInitializer.get_dynamic(
              task_params.values["initializer"]),
          activation=act_fn)(
              inp)

    datasets = self.datasets
    num_clases = self.num_clases

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = datasets

      def init(self, rng: PRNGKey) -> Params:
        init_net, unused_apply_net = hk.without_apply_rng(
            hk.transform(_forward))
        image = next(self.datasets.train)["image"]
        return init_net(rng, image)

      def loss(self, params: Params, rng: PRNGKey, data: Batch) -> jnp.ndarray:
        unused_init_net, apply_net = hk.without_apply_rng(
            hk.transform(_forward))

        image = data["image"]
        logits = apply_net(params, image)
        labels = jax.nn.one_hot(data["label"], num_clases)
        vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
        return jnp.mean(vec_loss)

      def normalizer(self, out):
        max_class = onp.log(2 * num_clases)
        out = jnp.nan_to_num(
            out, nan=max_class, neginf=max_class, posinf=max_class)
        return (jnp.clip(out, 0, max_class) -
                onp.log(num_clases / 5)) * 10 / onp.log(num_clases)

    return _Task()


@gin.configurable
def sample_image_mlp(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricImageMLP."""
  key1, key2, key3, key4, key5 = jax.random.split(key, 5)
  hidden_size = parametric_utils.log_int(key1, 8, 128)
  num_layers = parametric_utils.choice(key2, [0, 1, 2, 3, 4])
  image_size = parametric_utils.log_int(key3, 4, 32)
  batch_size = parametric_utils.log_int(key4, 4, 512)

  dataset_name = parametric_utils.SampleImageDataset.sample(key5)
  lf = cfgobject.LogFeature

  dataset = cfgobject.CFGObject(dataset_name, {
      "image_size": lf((image_size, image_size)),
      "batch_size": lf(batch_size),
  })
  num_classes = parametric_utils.SampleImageDataset.num_classes(dataset_name)

  return cfgobject.CFGObject(
      "ParametricImageMLP", {
          "hidden_layers": lf(num_layers * [hidden_size]),
          "num_classes": num_classes,
          "datasets": dataset
      })


@gin.configurable()
def timed_sample_image_mlp(key: PRNGKey, max_time=1e-5):
  model_path = model_paths.models[("sample_image_mlp", "time")]
  return time_model.rejection_sample(sample_image_mlp, model_path, key,
                                     max_time)


def sample_image_mlp_subsample_data(key: PRNGKey):
  rng = hk.PRNGSequence(key)
  cfg = sample_image_mlp(next(rng))
  ds = cfg.kwargs["datasets"]
  if "imagenet" not in ds.obj:
    if jax.random.uniform(next(rng), []) < 0.2:
      frac = float(jax.random.uniform(next(rng), []))
      cfg.kwargs["data_fraction"] = cfgobject.DoNotFeaturize(
          frac, ("time", "valid"))
  return cfg


@gin.configurable()
def timed_sample_image_mlp_subsample_data(key: PRNGKey, max_time=1e-5):
  model_path = model_paths.models[("sample_image_mlp", "time")]
  return time_model.rejection_sample(sample_image_mlp_subsample_data,
                                     model_path, key, max_time)
