from typing import Any, Optional

import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import functools
from utils.tree_util import *
from dataclasses import dataclass

PRNGKey = jnp.ndarray

@flax.struct.dataclass
class L2EState:
  params: Any
  momentum: Any
  direction: Any 
  noise: Any
  precond: Any
  rolling_features: common.MomAccumulator
  iteration: jnp.ndarray
  state: Any
  normal_key: Any


class L2E(lopt_base.LearnedOptimizer):
    def __init__(self,
                compute_summary=True,
                config=None):
        super().__init__()
        self._compute_summary = compute_summary

        def mlp_Q(inp):
            mlp = hk.nets.MLP([32, 32, 2])
            return mlp(inp)
        
        self.forward_Q = hk.without_apply_rng(hk.transform(mlp_Q))
        self.config = config

    def init(self, key: PRNGKey) -> lopt_base.MetaParams:
        input_Q = jnp.ones([1, 9])
        params_Q = self.forward_Q.init(key, input_Q)
        return params_Q

    def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
        net_Q = self.forward_Q
        exp_mult = self.config.meta.exp_mult
        precond_mult = self.config.meta.precond_mult
        noise_mult = self.config.meta.noise_mult
        step_mult = self.config.meta.step_mult
        
        num_data = 40000
        burnin_step = 100*(num_data//128)
        cycle_steps = (num_data//128) * 50

        class _Opt(opt_base.Optimizer):
            def __init__(self, theta: lopt_base.MetaParams):
                super().__init__()
                self.theta = theta
                self.decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

            def init(self,
                    params: lopt_base.Params,
                    model_state: Any = None,
                    num_steps: Optional[int] = None,
                    key: Optional[PRNGKey] = None) -> L2EState:
                return L2EState(
                        params = params,
                        momentum = jax.tree_util.tree_map(jnp.zeros_like, params),
                        rolling_features = common.vec_rolling_mom(self.decays).init(params),
                        iteration = jnp.asarray(0, dtype=jnp.int32),
                        direction = jax.tree_util.tree_map(jnp.zeros_like,params),
                        noise = jax.tree_util.tree_map(jnp.zeros_like,params),
                        precond = jax.tree_util.tree_map(jnp.zeros_like,params),
                        state = model_state,
                        normal_key = jax.random.PRNGKey(0)
                        )

            def update(self,
                    opt_state: L2EState,
                    grad: Any,
                    loss: float,
                    model_state: Optional[opt_base.ModelState] = None,
                    is_valid: bool = False,
                    key: Optional[PRNGKey] = None) -> L2EState:
                
                meta_params = self.theta
                params = opt_state.params
                momentum = opt_state.momentum
                normal_key = opt_state.normal_key

                next_rolling_features = common.vec_rolling_mom(self.decays).update(
                    opt_state.rolling_features, grad)

                def schedule(step):
                    t = step
                    t = (t % cycle_steps) / cycle_steps
                    return 0.5 * (1 + jnp.cos(t * jnp.pi))

                def _second_moment_normalizer(x, axis, eps=1e-5): # to make input norm to 1
                    return x * jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))

                def _tanh_embedding(iterations):
                    f32 = jnp.float32

                    def one_freq(timescale):
                        return jnp.tanh(iterations / (f32(timescale)) - 1.0)

                    timescales = jnp.asarray(
                        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
                        dtype=jnp.float32)
                    return jax.vmap(one_freq)(timescales)

                def _update_tensor(p, g, m , mom):

                    inps = []

                    # feature consisting of raw gradient values
                    batch_g = jnp.expand_dims(g, axis=-1)
                    inps.append(batch_g)

                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of raw momentum values
                    batch_mom = jnp.expand_dims(mom, axis=-1)
                    inps.append(batch_mom)

                    # feature consisting of all momentum values
                    # batch_m = jnp.expand_dims(m, axis=-1)
                    inps.append(m)

                    inp_stack = jnp.concatenate(inps, axis=-1)
                    axis = list(range(len(p.shape)))

                    inp_stack = _second_moment_normalizer(inp_stack, axis=axis)
                    inp = inp_stack
                    # apply the per parameter MLP.
                    output = net_Q.apply(meta_params, inp)
                    add_term = output[..., 0]
                    precond_term = output[..., 1] 
                    out = jnp.stack([add_term, precond_term],axis=0)

                    return out

                out = jax.tree_util.tree_map(_update_tensor, opt_state.params,
                                                    grad, next_rolling_features.m , opt_state.momentum)
                add_term = jax.tree_util.tree_map(lambda x: exp_mult * x[0] , out)
                precond_term = jax.tree_util.tree_map(lambda x: precond_mult * x[1], out) # precond_mult
     
                eps = jnp.sqrt(step_mult / 50000) 
                
                a = 0.1 
                grad_mom = precond_term #

                add_term2 = tree_mult(a, grad_mom)
                noise, normal_key = normal_like_tree(params , normal_key)
                noise = jax.tree_util.tree_map(lambda x : jnp.sqrt(2. * a * noise_mult )* x , noise)
                momentum = jax.tree_util.tree_map(lambda m, g, add, add2 , n : m - eps * (50000 * g + add + add2) + n ,
                                                    momentum, grad , add_term, add_term2 , noise)

                out = jax.tree_util.tree_map(_update_tensor, opt_state.params,
                                                    grad, next_rolling_features.m , momentum)
                grad_mom = jax.tree_util.tree_map(lambda x: precond_mult * x[1], out)                                   

                params = jax.tree_util.tree_map(lambda p, q: p + eps * q, params, grad_mom)

                next_opt_state = L2EState(
                    params = params, 
                    momentum = momentum,
                    rolling_features= next_rolling_features, 
                    iteration=opt_state.iteration + 1,
                    direction = jax.tree_util.tree_map(lambda x: x[0], out),
                    noise = grad_mom, 
                    precond = jax.tree_util.tree_map(lambda x: x[2], out),
                    state = model_state,
                    normal_key = normal_key
                    )   
                return next_opt_state

        return _Opt(theta)
