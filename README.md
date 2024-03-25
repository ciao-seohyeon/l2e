
# Learning to Explore for Stochastic Gradient MCMC

This repository contains code for creating the results for the ICML 2024 submission *"Learning to Explore for Stochastic Gradient MCMC"*.


## Getting Started
---- 
As this code is based on Google's learned optimizer library, please setting the environment following the instruction of [learned-optimization](https://github.com/google/learned_optimization).

For Tiny-ImageNet, you have to download npy files and place it to the directory ```./dataset```.

## Usage
----
1. First, execute the ```meta_training.py``` code to train the meta learner.
2. Use the trained meta learner checkpoint to run ```evaluation.py``` and collect samples.
3. Analyze the results using the collected samples.

## Meta Training
----
To meta train L2E, execute the following command:
```
python meta_training.py --config_file=./configs/meta.yaml --train_log_dir=./result/meta_training
```

## Evaluation
---- 
To train and evaluate a model using the learned L2E:
```
python evaluation.py --config_file=./configs/c10_frn.yaml --train_log_dir=./result/eval
```
To train ResNet20-FRN architecture with Swish activations on CIFAR-10 data, use ```./configs/c10_frn.yaml```. For CIFAR-100 data, use ```./configs/c100_frn.yaml```. To train ResNet56-FRN with Swish activation on Tiny ImageNet data, utilize ```./configs/tiny_frn.yaml```.

## Metric
----
For convergence analysis, please refer to ```./diagnosis.ipynb```. We used models trained with three different seeds to calculate $\hat{R}$ [(Gelman & Rubin, 1992)](https://projecteuclid.org/journals/statistical-science/volume-7/issue-4/Inference-from-Iterative-Simulation-Using-Multiple-Sequences/10.1214/ss/1177011136.full) and ESS.