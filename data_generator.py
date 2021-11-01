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
                       CSV_COLUMNS = ['x'],
                       batch_size=100,
                       num_epochs=1):

    file_path = os.path.join(filepath, filename)
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=batch_size,
                                                    column_names=CSV_COLUMNS,
                                                    shuffle=False,
                                                    num_epochs=num_epochs)
    return dataset


if __name__ == "__main__":
    data = uniform_generator(range=[-1.0, 1.0])
    print(data)
    save_data_to_csv(data=data)

    # dataset = read_data_from_csv()
    # ds = iter(dataset)
    # inputs = ds.get_next()
    # inputs['x']
