import functools
from typing import Any, Mapping

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import resnet
from learned_optimization.tasks.datasets import image

from tasks.resnet20_frn import ResNet20_frn, ResNet56_frn

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


args = [
        'blocks_per_group', 'use_projection', 'channels_per_group',
        'initial_conv_kernel_size', 'initial_conv_stride', 'max_pool',
        'resnet_v2', 'bottleneck'
    ]


def _fc_resnet_loss_fn_tmp(num_classes: int=10):
    
    def _fn(batch):
        net = ResNet20(num_classes=num_classes, bn_config={'decay_rate': 0.9})
        logits = net(batch['image'], is_training=True)
        loss = base.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes)
        )
        return jnp.mean(loss), logits

    return _fn

def _fc_resnet_loss_fn_frn(num_classes: int=10):
    
    def _fn(batch):
        net = ResNet20_frn(num_classes=num_classes, bn_config={'decay_rate': 0.9})
        logits = net(batch['image'], is_training=True)
        loss = base.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes)
        )
        
        return jnp.mean(loss), logits

    return _fn

def _fc_resnet_loss_fn_frn56(num_classes: int=10):
    
    def _fn(batch):
        net = ResNet56_frn(num_classes=num_classes, bn_config={'decay_rate': 0.9})
        logits = net(batch['image'], is_training=True)
        loss = base.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes)
        )
        
        return jnp.mean(loss), logits

    return _fn


class _ResNetTask(base.Task):

    def __init__(self,
                 base_model_fn,
                 datasets,
                 weight_decay):
        
        self.datasets = datasets
        self.weight_decay = weight_decay
        self._mod = hk.transform_with_state(base_model_fn) 

    def init(self, key) -> Params:
        params, unused_state = self.init_with_state(key)
        return params

    def init_with_state(self, key: chex.PRNGKey) -> base.Params:
        batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                         self.datasets.abstract_batch)
        return self._mod.init(key, batch)

    def loss(self, params, key, data):
        loss, _, _ = self.loss_with_state(params, None, key, data)
        l2_norm = 0.5 * self.weight_decay * sum([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)])
        return jnp.mean(loss) + l2_norm
    
    def logit(self, params, key, data):
        loss, logits, _ = self.loss_with_state(params, None, key, data)
        return logits

    def batch_nll(self, params, key, data):
        loss, logits, _ = self.loss_with_state(params, None, key, data)
        return logits, loss
    
    def nll(self, params, key, data):
        nll1, _, _ = self.loss_with_state(params, None, key, data)
        return nll1

    def loss_with_state(self, params, state, key, data):
        loss, logits, state, _ = self.loss_with_state_and_aux(params, state, key, data)
        return loss, logits, state

    def loss_with_state_and_aux(self, params, state, key, data):
        (loss, logits), state = self._mod.apply(params, state, key, data)
        return loss, logits, state, {}
