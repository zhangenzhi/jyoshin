import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.array_ops import repeat


def uniform_generator(range, n=1000000):
    x = tf.random.uniform(shape=[n], minval=range[0],
                          maxval=range[1], seed=9527)
    return x


def save_data_to_csv(data, filename='uniform.csv', filepath='./'):
    file = os.path.join(filepath, filename)
    np_data = data.numpy()
    np.savetxt(file, np_data, header='x', comments="")


def read_data_from_csv(filename='uniform.csv',
                       filepath='./',
                       CSV_COLUMNS=['x'],
                       batch_size=100,
                       num_epochs=1):

    file_path = os.path.join(filepath, filename)
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=batch_size,
                                                    column_names=CSV_COLUMNS,
                                                    shuffle=False,
                                                    num_epochs=num_epochs)
    return dataset


def read_data_from_cifar10(filepath='./',
                           shuffle=False,
                           batch_size=32,
                           num_epochs=1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        {'x': x_train, 'y': y_train})
    train_dataset = train_dataset.map(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).repeat(num_epochs)
    if shuffle:
        train_dataset = train_dataset.shuffle(50000)

    return train_dataset


if __name__ == "__main__":
    # data = uniform_generator(range=[-1.0, 1.0])
    # print(data)
    # save_data_to_csv(data=data)
    # read_data_from_cifar10()
    ds = read_data_from_csv(filename='labeled.csv',
                       filepath="./dataset/image/cifar10", 
                       CSV_COLUMNS=['y'],
                       batch_size=128)

    # i_ds = iter(ds)
    # x = []
    # with tf.device("/device:gpu:0"):
    #     for _ in range(100):
    #         v = i_ds.get_next()
    #         x.append(v['y'])
    #     x_gpu = tf.concat(x, axis=0)
    # print(x_gpu.device)
    
    # import pdb
    # pdb.set_trace()
        

    # dataset = read_data_from_csv()
    # ds = iter(dataset)
    # inputs = ds.get_next()
    # inputs['x']
