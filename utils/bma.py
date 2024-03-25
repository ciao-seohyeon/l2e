import jax
import jax.numpy as jnp
import numpy as onp

from scipy.special import logsumexp
from utils.metric import *
from collections import defaultdict

def BMA(task, key, param_list, test_indices, test_ds):
    
    def make_predictions(*args, **kwargs):
        true_labels, pred_lconfs = [], []
        total_logit, total_nll, BMA_acc, BMA_ece, BMA_nll = [], [], [], [], []
        single_acc = defaultdict(list)
        single_nll = defaultdict(list)
        K = len(param_list)
        for batch_indices in test_indices:
            logits, nlls = [], [] 
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], test_ds)
            true_labels.append(batch['label'])
            for i in range(K):
                params = param_list[i]
                logits_ = jax.nn.log_softmax(task.logit(params, key, batch))
                nll = task.nll(params, key, batch)
                logits.append(logits_)
                nlls.append(nll)
                single_acc[f'acc_{i}'].append((logits_.argmax(-1) == batch['label']).mean())
                single_nll[f'nll_{i}'].append(nll.mean())

            ens_logit = jnp.stack(logits, 1)
            avg_logits = logsumexp(jnp.stack(logits, 0), 0) - jnp.log(K)
            labels = jax.nn.one_hot(batch['label'], avg_logits.shape[-1])
            nll = jnp.mean(-jnp.sum(avg_logits * labels,-1), -1)
            acc = (avg_logits.argmax(-1) == batch['label']).mean()
            BMA_acc.append(acc)
            BMA_nll.append(nll)            
            total_logit.append(ens_logit)
        
        total_logit = jnp.concatenate(total_logit)
        true_labels = jnp.concatenate(true_labels)
        BMA_mean_acc = onp.mean(BMA_acc)
        BMA_mean_nll = onp.mean(BMA_nll)
        individual_acc = [onp.mean(v) for k,v in single_acc.items()]
        individual_nll = [onp.mean(v) for k,v in single_nll.items()]
        return true_labels, total_logit, individual_acc, individual_nll, BMA_mean_acc, BMA_mean_nll
    
    true_labels, total_logit, BMA_acc, BMA_nll, BMA_mean_acc, BMA_mean_nll = make_predictions()

    avg_logit = logsumexp(total_logit,1) - jnp.log(len(param_list))
    BMA_ece = evaluate_ece(avg_logit, true_labels)['ece']
    BMA_agr = - jnp.mean(jnp.sum(jnp.exp(avg_logit)*avg_logit, -1))
    BMA_kld = compute_pairwise_kld(total_logit)

    return BMA_mean_acc, BMA_mean_nll, BMA_acc, BMA_nll, BMA_ece, BMA_agr, BMA_kld

