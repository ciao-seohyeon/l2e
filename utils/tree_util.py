import jax

def tree_multiply(a,b):
  return jax.tree_util.tree_map(lambda e1, e2: e1*e2, a, b) 
   
def tree_mult(a,b): # 상수곱
  return jax.tree_util.tree_map(lambda x: x*a,b)

def tree_add(a, b):
  return jax.tree_util.tree_map(lambda e1, e2: e1+e2, a, b)

def normal_like_tree(a, key, std=1.0):
    treedef = jax.tree_util.tree_structure(a)
    num_vars = len(jax.tree_util.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_util.tree_map(lambda p, k: std*jax.random.normal(k, shape=p.shape),
            a, jax.tree_util.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]
