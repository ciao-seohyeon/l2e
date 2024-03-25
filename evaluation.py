from typing import Optional, Sequence
from datetime import datetime
import optax
import os
import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pylab as plt

import pickle
import logging
import argparse
import yaml
import nxcl
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict

from jax.scipy.special import logsumexp
from learned_optimization import filesystem
from learned_optimization import summary
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
from learned_optimization.optimizers import opt_to_optax
from learned_optimization.optimizers import optax_opts
from learned_optimization import tree_utils
import time
from utils.parse import *
from utils.tree_util import *
from utils.bma import BMA
from utils.loss_surface import loss_surface
from utils.metadata import get_metadata
from utils.metadata import *

from functools import partial
import matplotlib

now = datetime.now()
now_time = f'{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}'

class MetaMCMC:
    def __init__(self, config, num_data, lopt):
        self.lr = config.eval.init_lr
        self.alpha = config.eval.alpha
        self.num_data = num_data
        self.burnin_epochs = config.eval.burn_in
        self.cycle_epochs = config.eval.cycle_epochs
        self.thin = config.eval.thin
        self.batch_size = config.eval.batch_size
        self.temperature = 1.0
        self.lopt = lopt
        
    def _init_state(self,task, key):
        params = task.init(key) 
        _, normal_key = jax.random.split(key)
        momentum = jax.tree_util.tree_map(jnp.zeros_like, params)
        t_curr = jnp.asarray(0)
        return (params, momentum, t_curr, normal_key) 

    @partial(jax.jit, static_argnums=(0,))
    def update(self, opt_states, grads, loss):
        burnin_step = self.burnin_epochs * (self.num_data // self.batch_size)
        
        opt_state, params, momentum, t_curr, normal_key = opt_states
        eps = jnp.sqrt(self.lr / self.num_data)
        alpha = self.alpha
        _ , normal_key = normal_like_tree(params, normal_key)                

        def _burnin(opt_state, params, momentum):
          momentum = jax.tree_util.tree_map(lambda m, g : (1 - alpha) * m - eps * self.num_data * g , momentum, grads)
          update = jax.tree_util.tree_map(lambda x: x * eps, momentum)
          params = tree_add(params, update)
          return opt_state, params, momentum

        def _expl(opt_state, params, momentum):
          updates, opt_state = self.lopt.update(grads, opt_state, params=params, extra_args={"loss": loss}) 
          params = optax.apply_updates(params, updates)
          return opt_state, params, momentum

        opt_state, params, momentum = _expl(opt_state,params,momentum)
        state = opt_state, params, momentum, t_curr+1, normal_key
        return state
    
    
def eval(summary_writer, lopt, meta_params, config):
  metadata = get_metadata(config)
  num_data = metadata['num_train']
  num_steps = num_data // config.eval.batch_size
  num_test_steps = metadata['num_test'] // config.eval.batch_size
  max_iter = num_steps * config.eval.num_epoch

  # initialization
  print(f'Evaluation task: {config.eval.task}')
  task = parse_eval_task(config.eval.task, config.eval.batch_size , config)
  grad_fn = jax.jit(jax.value_and_grad(task.loss))
  key = jax.random.PRNGKey(config.eval.eval_seed) # jax.random.PRNGKey(0) 
  params = task.init(key)
  momentum = jax.tree_util.tree_map(jnp.zeros_like, params)

  lo_opt = lopt.opt_fn(meta_params)
  lo_opt = opt_to_optax.opt_to_optax_opt(lo_opt, num_steps=num_steps)
  opt_state = lo_opt.init(params)
  
  # get optimizer
  opt = MetaMCMC(config, num_data=metadata['num_train'], lopt=lo_opt)
  params, momentum, t_curr, normal_key = opt._init_state(task, key)
  opt_states = opt_state, params, momentum, t_curr, normal_key
  
  # upload eval task 
  train_ds = get_data(config, metadata, split='train')
  train_ds = {'image':train_ds[0], 'label':train_ds[1]}
  test_ds = get_data(config, metadata, split = 'test')
  test_ds = {'image':test_ds[0], 'label':test_ds[1]}
  key_train = jax.random.PRNGKey(config.eval.eval_seed)
  key_test = jax.random.PRNGKey(config.eval.eval_seed)

  @jax.jit
  def short_segment(opt_state, seq_of_indices, train_ds):
      def step(opt_state, batch_indices):
          batch = jax.tree_map(lambda x: x[batch_indices], train_ds)
          loss, grads = grad_fn(opt_state[1], key, batch)
          nll = task.nll(opt_state[1], key, batch)
          opt_state = opt.update(opt_state, grads, loss)
          return opt_state, (loss, nll)
      opt_state, (losses, nlls) = jax.lax.scan(step, opt_state, seq_of_indices)
      return opt_state, (losses, nlls)
  
  @jax.jit
  def short_segment_test(opt_state, seq_of_indices, test_ds):
      def step_test(opt_state, batch_indices):
          batch = jax.tree_map(lambda x: x[batch_indices], test_ds)
          loss, grads = grad_fn(opt_state[1], key, batch)
          nll = task.nll(opt_state[1], key, batch)
          return opt_state, (loss, nll)
      opt_state, (losses, nlls) = jax.lax.scan(step_test, opt_state, seq_of_indices)
      return opt_state, (losses, nlls)
    
  train_step = partial(short_segment, train_ds=train_ds)
  test_step = partial(short_segment_test, test_ds=test_ds)

  test_indices, _ =  get_indices(metadata, num_test_steps, config.eval.batch_size, key_test, split='test')

  # training loop
  def is_ensembling_epoch(config, epoch):
    if epoch <= config.eval.burn_in:
        return False
    return (epoch % config.eval.thin == 0)
  save_dir = f'{config.train_log_dir}/{now_time}'

  losses, nlls = [], []
  param_list = []
  t_curr = 0
  
  for epoch in tqdm.trange(1, config.eval.num_epoch+1):    
    train_indices, key_train = get_indices(metadata, num_steps, config.eval.batch_size, key_train, split='train')
    start= time.time()
    opt_states, (l, nll) = train_step(opt_states, train_indices)
    end = time.time()
    opt_state, params, momentum, t_curr, _ = opt_states
    losses.append(jnp.mean(l))
    nlls.append(jnp.mean(nll))
    
    if is_ensembling_epoch(config, epoch):
      logging.info(f'[{epoch}/{config.eval.num_epoch}] \t save model ...')
      param_list.append(params)

    summary_writer.scalar("loss", np.mean(losses), step=epoch)
    summary_writer.scalar("nll", np.mean(nlls), step=epoch)
    summary_writer.flush()


  save_dir = f'{config.train_log_dir}/{now_time}'

  # Plot loss and nll
  _, axes = plt.subplots(1, 2, figsize=(10, 5))
  axes[0].plot(losses, label='loss')
  axes[1].plot(nlls, label='nll')
  axes[0].legend()
  axes[1].legend()
  plt.savefig(f'{save_dir}/loss_nll.png')
  plt.close()
  
  
  with open(f'{save_dir}/meta_params.pickle', 'wb') as f:
      pickle.dump(meta_params, f, pickle.HIGHEST_PROTOCOL)

  
  print(f'Number of BMA models: {len(param_list)}')
  logging.info(f'Number of BMA models: {len(param_list)}')
  results = BMA(task, key, param_list, test_indices, test_ds)
  loss_surface(test_ds, task, param_list, save_dir)
  
  with open(f'{save_dir}/model.pickle', 'wb') as f:
    pickle.dump(param_list, f, pickle.HIGHEST_PROTOCOL)

  return results


def main(config):
  np.random.seed(config.eval.eval_seed)
  # make train log directory
  train_dir = f'{config.train_log_dir}/{now_time}'
  print(f'Train directory: {train_dir}')
  filesystem.make_dirs(train_dir)

  # logging
  setup_logging(train_dir)
  logging.info('Setup experiments! Training directory: %s', train_dir)
  save_config(config, os.path.join(train_dir, "config.yaml"))
  summary_writer = summary.MultiWriter(summary.JaxboardWriter(train_dir))
  
  # load meta params, lopt
  with open(f'{config.eval.lopt_path}/meta_params.pickle', 'rb') as f:
    meta_params = pickle.load(f)
  lopt_config = load_config(config.eval.lopt_path+'/config.yaml')
  lopt_config.meta.step_mult = config.eval.step_mult 
  lopt = parse_lopt(lopt_config.train.lopt, lopt_config)
  
  # evaluation
  BMA_acc, BMA_nll, single_acc, single_nll, BMA_ece, BMA_agr, BMA_kld = eval(summary_writer, lopt, meta_params, config)
  print(f'BMA acc: {BMA_acc}, BMA nll: {BMA_nll}, BMA ece: {BMA_ece}, BMA agreement: {BMA_agr}, BMA KLD, {BMA_kld}')
  logging.info(f'BMA acc: {BMA_acc}, BMA nll: {BMA_nll}, BMA ece: {BMA_ece}, BMA agreement: {BMA_agr}, BMA KLD, {BMA_kld}')
  logging.info(f'Single ACC: {single_acc}')
  logging.info(f'Single NLL: {single_nll}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Meta Learning')
  parser.add_argument('-f', '--config_file', type=str, required=True)
  args, rest_args = parser.parse_known_args()

  config: ConfigDict = load_config(args.config_file)
  parser = argparse.ArgumentParser()
  add_config_arguments(parser, config, aliases={
      # evaluation
      "eval.seed":                  ["--train_seed"],
      "eval.task":                  ["--eval_task"],
      "eval.batch_size":            ["--batch_size"],
      "eval.num_epoch":             ["--num_epoch"],
      "eval.thin":                  ["--thin"],
      "eval.burn_in":               ["--burn_in"],
      "eval.alpha":                 ["--alpha"],
      "eval.init_lr":               ["--init_lr"],
      "eval.cycle_epochs":          ["--cycle_epochs"],
      "eval.lopt_path":             ["--lopt_path"],
      "eval.nll_lopt_path":         ["--nll_lopt_path"],
      "eval.step_mult":             ["--step_mult"]
  })
  parser.add_argument('--train_log_dir', type=str, help='train log directory', required=True)
  args = parser.parse_args(rest_args)
  config.update(vars(args))

  main(config)

