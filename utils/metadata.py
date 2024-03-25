import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

def get_metadata(config):
    metadata = {}
    if config.eval.task in ['mnist', 'fmnist', 'fmnist_conv', 'emnist']: 
        metadata['num_train'] = 48000 
        metadata['num_valid'] = 12000
        metadata['num_test'] =  10000
        metadata['num_classes'] = 10
        metadata['shape'] = (28, 28, 1)
    elif config.eval.task in ['c10_frn','c100_frn']:
        metadata['num_train'] = 40960
        metadata['num_valid'] = 10000
        metadata['num_test'] =  10000
        metadata['num_classes'] = 10 if config.eval.task == 'cifar' else 100 
        metadata['shape'] = (32, 32, 3)
    else:
        raise ValueError(f'Invalid data {config.eval.task}')

    metadata['num_train_batches'] = metadata['num_train'] // config.eval.batch_size
    metadata['num_test_batches'] = metadata['num_test'] // config.eval.batch_size

    return metadata

   
def get_nist(name, split='train', num_data=None , subset=False):
    
    ds = tfds.load(name,
            split=split if num_data is None else f'{split}[:{num_data}]',
            as_supervised=True)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def filter_labels(image, label):
        return tf.math.logical_and(label >= 0, label <= 5)
    if subset:
        ds = ds.filter(filter_labels)
    else:
        pass
    ds = ds.map(preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    ds = ds.batch(num_data or 60000)
    ds = tfds.as_numpy(ds)

    return jax.tree_util.tree_map(jnp.asarray, next(iter(ds)))


def get_cifar(name:str, split:str,  data_augmentation:int=0, num_data=None):
    
    ds,ds_info = tfds.load(name,
            split=(split if num_data is None else f'{split}[:{num_data}]'),
            as_supervised=True,
            with_info = True)
    
    image_shape = ds_info.features['image'].shape
    
    def preprocess(image, label):
      image_mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
      image_std =  tf.constant([[[0.2470, 0.2435, 0.2616]]])

      if data_augmentation==1:
          image = tf.pad(image, [[2, 2], [2, 2], [0, 0]],'CONSTANT') 
          image = tf.image.random_crop(image, image_shape)
          image = tf.image.random_flip_left_right(image)
      else:
          pass

      image = tf.cast(image, tf.float32) / 255.0 
      image = (image - image_mean) / image_std

      return image, label

    ds = ds.map(preprocess)
    ds = ds.batch(60000) # 60000 
    ds = tfds.as_numpy(ds)

    return jax.tree_util.tree_map(jnp.asarray, next(iter(ds)))



def get_data(config, metadata, split='train'):
    if split == 'train':
        num_data = metadata['num_train']
    elif split == 'test':
        num_data = metadata['num_test']
    else:
        raise ValueError(f'Unknown split type ( valid is not currently available)')

    if config.eval.task in ['mnist', 'fmnist', 'fmnist_conv', 'emnist']:
        if config.eval.task in ['fmnist', 'fmnist_conv']:
            data = 'fashion_mnist'
            return get_nist(data, split=split, num_data=num_data)
        elif config.eval.task in ['mnist']:
            data = 'mnist'
            return get_nist(data, split=split, num_data=num_data)
        elif config.eval.task in ['emnist']:
            data = 'emnist'
            return get_nist(data, split=split, num_data=num_data)
    elif config.eval.task in ['c10_frn']:
        data = 'cifar10'
        return get_cifar(name=data, split=split, num_data=num_data)
    elif config.eval.task in ['c100_frn']:
        data = 'cifar100'
        return get_cifar(name=data, split=split, num_data=num_data)

    else:
        raise ValueError(f'Unknown dataset {config.eval.task}')


def get_indices(metadata:dict, num_step:int, 
                     batch_size:int, key:jnp.array, split:str='train'):
    
    '''
    get batch corresponding to 1 epoch
    
    output['image'] :  [num_step , batch_size , img_size]
    output['label'] :  [num_step, batch_size , label_num]
    '''
    if split == 'train':
        num_data = metadata['num_train']
    elif split == 'test':
        num_data = metadata['num_test']
    elif split == 'valid':
        num_data = metadata['num_valid']
    else:
        raise ValueError(f'Unknown split type ( valid is not currently available)')
    
    key, perm_key = jax.random.split(key, 2)
    indices = jax.random.permutation(perm_key,jnp.arange(int(num_step*batch_size)))
    indices = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (num_step,batch_size)),
            indices)
    
    return indices, perm_key