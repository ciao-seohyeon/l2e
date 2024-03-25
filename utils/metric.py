import jax
import jax.numpy as jnp
import numpy as onp

from collections import defaultdict

# import numpy as jnp
## 
def _flatten_batch_axes(arr):
    pred_dim = arr.shape[-1]
    batch_dim = arr.size // pred_dim
    return arr.reshape((batch_dim, pred_dim))


def _flatten_outputs_labels(outputs, labels):
  return _flatten_batch_axes(outputs), labels.reshape(-1)

def MD_2(single_nll, bma_nll):
    # single_nll [b, ens_num]
    # bma_nll [b,]
    '''
    variance of functions among ensemble member
    \int(p(y|x,w) - p(y|x)) * p(w|D) dw  \approx 1/K *  sum(2 log(p_i / log p_ens ))
    '''
    MD = jnp.mean((jnp.exp(-single_nll) - jnp.expand_dims(jnp.exp(-bma_nll),-1))**2)
    return MD


## average over whole label space
def MD_2_logit(single_logit, bma_logit):
    # single_logit [ens_num , b, num_label]
    # bma_logit [b,num_label]
    prob1 = jax.nn.softmax(single_logit,-1)
    bma_prob = jnp.expand_dims(jax.nn.softmax(bma_logit,-1),0) # [ 1,b,num_label]
    MD = jnp.mean( jnp.mean(jnp.mean( (prob1 - bma_prob) **2 , 0) , -1) ) 
    return MD


def evaluate_ece(confidences, true_labels, log_input=True, eps=1e-8, num_bins=15):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        num_bins (int): Specifies the number of bins used by the historgram binning.
    Returns:
        A dictionary of components for expected calibration error.
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    max_confidences = jnp.max(jnp.exp(log_confidences), axis=1)
    max_pred_labels = jnp.argmax(log_confidences, axis=1)
    raw_accuracies = jnp.equal(max_pred_labels, true_labels)
    
    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[ 1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_frequencies = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = jnp.logical_and(max_confidences > bin_lower, max_confidences <= bin_upper)
        bin_frequencies.append(jnp.sum(in_bin))
        if bin_frequencies[-1] > 0:
            bin_accuracies.append(jnp.mean(raw_accuracies[in_bin]))
            bin_confidences.append(jnp.mean(max_confidences[in_bin]))
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)
    
    bin_accuracies = jnp.array(bin_accuracies)
    bin_confidences = jnp.array(bin_confidences)
    bin_frequencies = jnp.array(bin_frequencies)

    return {
        'bin_lowers': bin_lowers,
        'bin_uppers': bin_uppers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_frequencies': bin_frequencies,
        'ece': jnp.nansum(
            jnp.abs(
                bin_accuracies - bin_confidences
            ) * bin_frequencies / jnp.sum(bin_frequencies)
        ),
    }
        
def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
    x = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def compute_pairwise_kld(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, M, K,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.
    Returns:
        An array of pairwise KL divergence (averaged over off-diagonal elements) with
        shape [1,] when reduction in ["mean",], or raw pairwise KL divergence values
        (per example) with shape [N, M, M,] when reduction in ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    n_datapoint = log_confidences.shape[0]
    n_ensembles = log_confidences.shape[1]
    raw_results = jnp.array([
        jnp.sum(
            jnp.multiply(
                jnp.exp(log_confidences[:, idx, :]),
                log_confidences[:, idx, :] - log_confidences[:, jdx, :],
            ), axis=1,
        ) for idx in range(n_ensembles) for jdx in range(n_ensembles)
    ]).reshape(n_ensembles, n_ensembles, n_datapoint).transpose(2, 0, 1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.sum(jnp.zeros(1)) if n_ensembles == 1 else jnp.sum(
            jnp.mean(raw_results, axis=0)
        ) / (n_ensembles**2 - n_ensembles)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def compute_pairwise_con(param_list, task, key, test_indices, test_ds, lamb=0.5):
    K = len(param_list)

    def acc(param):
        def _acc(param, test_indices):
            batch = jax.tree_util.tree_map(lambda x: x[test_indices], test_ds)
            logits_ = jax.nn.log_softmax(task.logit(param, key, batch))
            corrects = (logits_.argmax(-1) == batch['label']).mean()
            return corrects

        accs = jax.vmap(_acc, in_axes=(None, 0))(param, test_indices)
        return accs.mean().item()

    q_pair = []
    acc_tmp = 0.0
    lamb_dir = jax.random.dirichlet(key, jnp.ones(K))
    for i in range(K):
        param1 = param_list[i]
        # pairwise
        if i != K-1:
            param2 = param_list[i+1]
            param = jax.tree_util.tree_map(lambda x, y: lamb * x + (1-lamb) * y, param1, param2)
            q_pair_i = acc(param) - (lamb * acc(param1) + (1-lamb) * acc(param2))
            q_pair.append(round(q_pair_i, 6))
        # joint
        if i == 0:
            params = param_list[0]
        else:
            params = jax.tree_util.tree_map(lambda x, y: x + lamb_dir[i] * y, params, param1)
        acc_tmp += lamb * acc(param1)
    q_joint = round(acc(params) - acc_tmp, 6)
    
    return q_pair, q_joint

def compute_pairwise_agr(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, M, K,].
        log_input (bool, unused): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.
    Returns:
        An array of pairwise agreement (averaged over off-diagonal elements) with
        shape [1,] when reduction in ["mean",], or raw pairwise agreement values
        (per example) with shape [N, M, M,] when reduction in ["none",].
    """
    pred_labels = jnp.argmax(confidences, axis=2) # [N, M,]
    pred_labels = onehot(pred_labels, confidences.shape[2]) # [N, M, K,]
    n_datapoint = pred_labels.shape[0]
    n_ensembles = pred_labels.shape[1]
    raw_results = jnp.array([
        jnp.sum(
            jnp.multiply(
                pred_labels[:, idx, :],
                pred_labels[:, jdx, :],
            ), axis=1,
        ) for idx in range(n_ensembles) for jdx in range(n_ensembles)
    ]).reshape(n_ensembles, n_ensembles, n_datapoint).transpose(2, 0, 1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.sum(jnp.ones(1)) if n_ensembles == 1 else (
            jnp.sum(jnp.mean(raw_results, axis=0)) - n_ensembles
        ) / (n_ensembles**2 - n_ensembles)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def compute_pairwise_cka(output_vecs, reduction="mean"):
    """
    Args:
        output_vecs (Array): An array with shape [N, M, K,].
        reduction (str): Specifies the reduction to apply to the output.
    Returns:
        An array of pairwise centered kernel alignment (averaged over off-diagonal elements) with
        shape [1,] when reduction in ["mean",], or raw pairwise centered kernel alignment values
        with shape [M, M,] when reduction in ["none",].
    """
    n_datapoint = output_vecs.shape[0]
    n_ensembles = output_vecs.shape[1]

    raw_results = []
    for idx in range(n_ensembles):
        for jdx in range(n_ensembles):
            identity_mat = jnp.diag(jnp.ones(n_datapoint))
            centering_mat = identity_mat - jnp.ones((n_datapoint, n_datapoint)) / n_datapoint
            x = output_vecs[:, idx, :]
            y = output_vecs[:, jdx, :]
            cov_xy = jnp.trace(
                x @ jnp.transpose(x) @ centering_mat @ y @ jnp.transpose(y) @ centering_mat
            )/ jnp.power(n_datapoint - 1, 2)
            cov_xx = jnp.trace(
                x @ jnp.transpose(x) @ centering_mat @ x @ jnp.transpose(x) @ centering_mat
            )/ jnp.power(n_datapoint - 1, 2)
            cov_yy = jnp.trace(
                y @ jnp.transpose(y) @ centering_mat @ y @ jnp.transpose(y) @ centering_mat
            )/ jnp.power(n_datapoint - 1, 2)
            raw_results.append(cov_xy / jnp.sqrt(cov_xx * cov_yy))
    raw_results = jnp.array(raw_results).reshape(n_ensembles, n_ensembles)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.sum(raw_results) / (n_ensembles**2 - n_ensembles)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


