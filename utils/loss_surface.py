import jax.numpy as jnp
import jax
import numpy as np
from jax.scipy.special import logsumexp
from learned_optimization.tasks.datasets import image
from tqdm import tqdm
from datetime import datetime
from matplotlib import pylab as plt
from utils.tree_util import tree_multiply, tree_mult, tree_add, normal_like_tree

def loss_surface(test_ds, task, param_list, save_dir):
    
    K = len(param_list)
    theta_1 = param_list[0] #3 -3 0
    theta_2 = param_list[1] #4 -2 1
    theta_3 = param_list[2] #5 -1 2

    param = theta_1
    raram = theta_2
    qaram = theta_3
    unravel_pytree = jax.flatten_util.ravel_pytree(param)[1]
    
    w0 = jax.flatten_util.ravel_pytree(param)[0]
    w1 = jax.flatten_util.ravel_pytree(raram)[0]
    wh = (w1 + w0) / 2.0
    wm = jax.flatten_util.ravel_pytree(
        jax.tree_util.tree_map(lambda a, b, c: 0.25*a + 0.50*b + 0.25*c, param, qaram, raram))[0]

    v1 = (w0 - wh) / jnp.sqrt(jnp.sum(jnp.square(w0 - wh))) # ( jnp.sqrt(jnp.sum(jnp.square(w0 - wh))) , 0) < -- theta _1 ,  (0,0 ) < -- theta1, theta 2  중점 
    v2 = (wm - wh) / jnp.sqrt(jnp.sum(jnp.square(wm - wh)))
    P = lambda a, b: wh + a*v1 + b*v2
    # batch = next(task.datasets.test) # train , since train loss not converge to zero in our setting train or test just execute once

    batch = jax.tree_util.tree_map(lambda x: x[:128], test_ds) 
    # test_ds # {'image': test_ds[0][:128], 'label': test_ds[1][:128]}
    # jax.tree_util.tree_map(lambda x: x[batch_indices], train_ds)(test_ds['image'][:128] , test_ds['label'][:128])
    # print(batch.shape)
    
    ## change aa ,bb value by pre calculate v1 
    aa = np.linspace(-jnp.sqrt(jnp.sum(jnp.square(w0 - wh)))*2 , jnp.sqrt(jnp.sum(jnp.square(w0 - wh)))*2, 40) #204840 -> 격자 50개
    bb = np.linspace(- 4 * jnp.sqrt(jnp.sum(jnp.square(wm - wh))), 4 * jnp.sqrt(jnp.sum(jnp.square(wm - wh))) , 40)
    # aa = np.linspace(-40, 40 , 20) #  bb = np.linspace(-100, 100, 50)    
    # bb = np.linspace(-40, 40, 20)  # for 40
    X, Y = np.meshgrid(aa, bb)
    key = jax.random.PRNGKey(0)

    def evaluate_acc(logits,labels):
        a_max = jnp.argmax(logits , -1)
        return (a_max == labels).mean()

    Z = []

    for a in aa:
        for b in bb:
            # make predictions
            tst_logits = []
            tst_labels = []
            # if ('CIFAR' in config.eval.task) or ('ResNet' in config.eval.task):
            _logits = task.logit(unravel_pytree(P(a, b)), key, batch)
            # print(_logits.shape)
            # else:
            #     _logits = task._mod.apply(unravel_pytree(P(a, b)) ,key , batch['image'])
            _labels = batch['label']
            tst_logits.append(_logits)
            tst_labels.append(_labels)
            
            tst_logits = jnp.concatenate(tst_logits)
            # print(tst_logits.shape)
            tst_labels = jnp.concatenate(tst_labels)
            Z.append(float(evaluate_acc(jax.nn.log_softmax(tst_logits, axis=-1), tst_labels)))

    Z = np.array(Z).reshape(len(aa), len(bb)).T

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    contour = ax.contourf(X, Y, Z, cmap='magma', levels=12)
    fig.colorbar(contour, format='%.2f')
    plt.scatter( jnp.sqrt(jnp.sum(jnp.square(w0 - wh))) , 0   , marker= 'o' , label='init') # w0
    plt.scatter( - jnp.sqrt(jnp.sum(jnp.square(w0 - wh))) , 0 , marker = 'o' ,label='1st cycle') # w1
    plt.scatter(0 , 2 * jnp.sqrt(jnp.sum(jnp.square(wm - wh)))  , marker = 'o' , label='2nd cycle') # w2 - should debug
    plt.legend()
    plt.tight_layout()
    plt.title("loss surface of params per cycle")
    plt.savefig(f'{save_dir}/loss_surface.png')    
    
    return None


# 