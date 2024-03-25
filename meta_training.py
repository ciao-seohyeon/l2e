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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple learned optimizer training example using gradient estimator APIs."""
from typing import Optional, Sequence
from datetime import datetime
import optax
import os
import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pylab as plt

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

from outer_trainers import gradient_learner
from outer_trainers import lopt_truncated_step
from outer_trainers import truncation_schedule
from outer_trainers.full_es_function_bma import FullES_function_BMA

from learned_optimization.tasks import base as tasks_base
from learned_optimization.optimizers import opt_to_optax
from learned_optimization.optimizers import optax_opts
from learned_optimization import tree_utils
from learned_optimization.tasks.parametric.cfgobject import *

from utils.parse import *
from utils.tree_util import *
from utils.bma import BMA
from utils.loss_surface import loss_surface
from utils.metadata import get_metadata
import pickle
from utils.metadata import *
from functools import partial
from tasks.task import *
from outer_trainers.gradient_learner import GradientLearnerState
import random

now = datetime.now()
now_time = f'{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}'

def train(summary_writer, config, train_dir):
  """Train a learned optimizer!"""
  onp.random.seed(config.train.train_seed)
  
  lopt = parse_lopt(config.train.lopt, config)
  key = jax.random.PRNGKey(int(config.train.train_seed))
  theta_opt = opt_base.Adam(config.train.outer_learning_rate)
  theta_opt = opt_base.GradientnormClipOptimizer(opt = theta_opt , grad_clip=1)
  gradient_estimators = []
  dummy_params = lopt.init(jax.random.PRNGKey(10))

  total_task_family = mnist_fmnist_cnn_depth_tasklist()

  ind = random.randint(0,len(total_task_family)-1)
  task_family = total_task_family[ind]
  trunc_sched = truncation_schedule.NeverEndingTruncationSchedule()
  truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family=task_family,
      learned_opt=lopt,
      trunc_sched=trunc_sched,
      num_tasks=config.train.num_tasks,
      meta_loss_split='outer_valid',
      random_initial_iteration_offset=0)
  trunc_sched = truncation_schedule.ConstantTruncationSchedule(config.train.max_length)
  grad_est = FullES_function_BMA(truncated_step, trunc_sched)
  gradient_estimators.append(grad_est)

  del task_family,grad_est

  outer_trainer = gradient_learner.SingleMachineGradientLearner(
      lopt, gradient_estimators, theta_opt)
  
  def maybe_resample_grad_estimator(total_task_family , lopt):

    gradient_estimators = []
    ind = random.randint(0,len(total_task_family)-1)
    task_family = total_task_family[ind]
    del ind
    trunc_sched = truncation_schedule.NeverEndingTruncationSchedule()
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family=task_family,
      learned_opt=lopt,
      trunc_sched=trunc_sched,
      num_tasks=config.train.num_tasks,
      meta_loss_split='outer_valid',
      random_initial_iteration_offset= 0)
    trunc_sched = truncation_schedule.ConstantTruncationSchedule(config.train.max_length)
    grad_est = FullES_function_BMA(truncated_step, trunc_sched)

    gradient_estimators.append(grad_est)
    del grad_est
    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, theta_opt)
    return outer_trainer 
  
  outer_trainer_state = outer_trainer.init(key)

  losses = []
  total_losses = []
  valid_losses = []

  for i in tqdm.trange(config.train.outer_iterations):
    with_m = True if i % 10 == 0 else False
    key1, key = jax.random.split(key)
    outer_trainer_state, loss, valid_loss, metrics = outer_trainer.update(
        outer_trainer_state, key1, with_metrics=with_m)

    losses.append(loss)
    total_losses.append(loss)
    valid_losses.append(valid_loss)
    logging.info(f'iter: {i}, loss: {loss}')
    logging.info(f'iter: {i}, valid_loss: {valid_loss}')

    # log out summaries to tensorboard
    if with_m:
      summary_writer.scalar("average_meta_loss", np.mean(losses), step=i)
      losses = []
      for k, v in metrics.items():
        agg_type, metric_name = k.split("||")
        if agg_type == "collect":
          summary_writer.histogram(metric_name, v, step=i)
        else:
          summary_writer.scalar(metric_name, v, step=i)
      summary_writer.flush()

    if i % 100 == 0:
      with open(f'{train_dir}/meta_params_{i}.pickle', 'wb') as f:
          meta_params = outer_trainer.get_meta_params(outer_trainer_state)
          pickle.dump(meta_params, f, pickle.HIGHEST_PROTOCOL)
          logging.info(f'[{i}/{config.train.outer_iterations}] fine_tuned meta parameters are saved')

    logging.info('change task')
    meta_params = outer_trainer.get_meta_params(outer_trainer_state)
    del outer_trainer_state, outer_trainer # delete previous ouer_trainer
    outer_trainer = maybe_resample_grad_estimator(total_task_family , lopt)
    outer_trainer_state = outer_trainer.init(key)
    new_theta_opt_state = outer_trainer_state.gradient_learner_state.theta_opt_state.replace(params = meta_params)
    outer_trainer_state = outer_trainer_state.replace(gradient_learner_state = GradientLearnerState(new_theta_opt_state) )  
  
  # get meta-param
  meta_params = outer_trainer.get_meta_params(outer_trainer_state)
  return lopt, meta_params, total_losses, valid_losses


def main(config):
  # make train log directory
  train_dir = f'{config.train_log_dir}/{now_time}'
  print(f'train directory: {train_dir}')
  filesystem.make_dirs(train_dir)
  # set logging file
  setup_logging(train_dir)
  logging.info('Setup experiments! Training directory: %s', train_dir)
  # save config file
  save_config(config, os.path.join(train_dir, "config.yaml"))
  # tensorboard logging
  summary_writer = summary.MultiWriter(summary.JaxboardWriter(train_dir))

  # meta training
  lopt, meta_params, total_losses , valid_losses = train(summary_writer, config, train_dir)
  
  plt.plot(total_losses , label = 'train')
  plt.plot(valid_losses , label = 'valid')
  plt.legend()
  plt.savefig(f'{train_dir}/meta_loss.png')
  plt.close()
  
  with open(f'{train_dir}/meta_params.pickle', 'wb') as f:
      pickle.dump(meta_params, f, pickle.HIGHEST_PROTOCOL)
      print('meta parameters are saved')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Meta Learning')
  parser.add_argument('-f', '--config_file', type=str, required=False)
  args, rest_args = parser.parse_known_args()

  config: ConfigDict = load_config(args.config_file)
  parser = argparse.ArgumentParser()
  add_config_arguments(parser, config, aliases={
      # meta train
      "train.seed":                 ["--train_seed"],
      "train.lopt":                 ["--lopt"],
      "train.num_tasks":            ["--num_tasks"],
      "train.outer_iterations":     ["--outer_iterations"],
      "train.outer_learning_rate":  ["--outer_learning_rate"],
      "train.max_length":           ["--max_length"],
      "train.min_length":           ["--min_length"],
      # lopt
      "meta.exp_mult":              ["--exp_mult"],
      "meta.step_mult":             ["--step_mult"],
      "meta.noise_mult":            ["--noise_mult"],
  })
  parser.add_argument('--train_log_dir', type=str, help='train log directory', required=True)
  args = parser.parse_args(rest_args)
  config.update(vars(args))

  main(config)

  