import os
import numpy as np
import tensorflow as tf

from models import DNN
from data_generator import read_data_from_csv, read_data_from_cifar10
from utils import check_mkdir


def run_uniform():
    dnn = DNN(units=[64, 16, 1], activations=['tanh', 'tanh', 'tanh'])
    dataset = read_data_from_csv(filename="uniform.csv", filepath="./")
    dataset = iter(dataset)
    generate_label_for_data(dnn, dataset)


def generate_label_for_data(model, dataset, filename='./labeled.csv'):
    labeled_data = None
    flag = 1
    while True:
        try:
            x = dataset.get_next()
        except:
            print("run out of data.")
            break
        flag += 1
        print(flag)
        x['x'] = tf.reshape(x['x'], (-1, 1))
        y = model(x['x'])
        z = tf.concat([x['x'], y], axis=1).numpy()
        if isinstance(labeled_data, type(None)):
            labeled_data = z
        else:
            labeled_data = np.concatenate([labeled_data, z], axis=0)
    np.savetxt(filename, labeled_data, header='x,y',
               comments="", delimiter=',')


def generate_label_for_cifar10(model, dataset, path_to_file='./', filename='labeled.csv'):
    labeled_data = None
    flag = 0
    check_mkdir(path_to_file)
    filename = os.path.join(path_to_file, filename)
    while True:
        try:
            x = dataset.get_next()
        except:
            print("run out of data.")
            break
        flag += 1
        print(flag)
        y = model(x['x'])
        y = tf.squeeze(y)
        y = tf.argmax(y, -1)
        if isinstance(labeled_data, type(None)):
            labeled_data = y
        else:
            labeled_data = np.concatenate([labeled_data, y], axis=0)
    np.savetxt(filename, labeled_data, header='y',
               comments="", delimiter=',')
