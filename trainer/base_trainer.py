import os
import h5py
from typing import KeysView
import tensorflow as tf
import sys

from model.cnn import CNN
sys.path.append('..')

from model import DNN, ResNet
from utils import check_file, check_mkdir, print_error, print_green


class BaseTrainer:
    def __init__(self, args):
        self.args = args

        self.strategy =  self._build_envs()
        
        with self.strategy.scope():
            self.dataset = self._build_dataset(self.args['dataset'])
            self.loss = self._build_loss(self.args['loss'])
            self.metric = self._build_metric(self.args['metric'])
            self.optimizer = self._build_optimizer(self.args['optimizer'])
            self.model = self._build_model(self.args['model'])

    def _build_envs(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        for item in physical_devices:
            tf.config.experimental.set_memory_growth(item, True)
        
        if 'distribute' in self.args['others'].keys():
            strategy = tf.distribute.MirroredStrategy()
        else:  # use one device strategy
            strategy = tf.distribute.get_strategy()
        return strategy

    def _build_model(self, model_args):
        if model_args['name'] == 'DNN':
            model = DNN(units=model_args['units'],
                        activations=model_args['activations'])
        elif model_args['name'] == "ResNet50":
            model = ResNet(classes=10, 
                           model_type=model_args['name'])
        elif model_args['name'] == "CNN":
            model = CNN(kernels=model_args['kernels'],
                        filters=model_args['filters'],
                        tail_acts=model_args['tail_acts'],
                        classes=model_args['classes'])
        else:
            model = None
        return model

    def _build_metric(self, metric_args):
        metric = tf.keras.metrics.get(metric_args['name'])
        return metric

    def _build_loss(self, loss_args):
        loss = tf.keras.losses.get(loss_args['name'])
        return loss

    def _build_optimizer(self, optimizer_args):

        if type(optimizer_args['learning_rate']) != float:
            lr_schedule = tf.keras.optimizers.schedules.deserialize(
                config=optimizer_args['learning_rate'])
            optimizer_args['learning_rate'] = lr_schedule

        if optimizer_args['name'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=optimizer_args['learning_rate'])
        else:
            try:
                optimizer = tf.keras.optimizers.get(optimizer_args['name'])
            except:
                print_error("no such optimizer")
        return optimizer

    def save_model_weights(self, filepath='./saved_models', name='latest.h5', save_format="h5"):
        filepath = os.path.join(filepath)
        check_mkdir(filepath)
        filepath = os.path.join(filepath, name)
        self.model.save_weights(filepath, save_format=save_format)
        print_green("model saved in {}".format(filepath))

    def load_model_weights(self, filepath='./saved_models', name='latest.h5'):
        filepath = os.path.join(filepath, name)
        if check_file(filepath):
            self._just_build()
            self.model.load_weights(filepath)
            print_green("model load from {}".format(filepath))
        else:
            print_error("file doesn't exits in {}.".format(filepath))
            exit()
    
    def save_weights_trajectory(self, filepath, filename='trail.h5'):
        save_to_hdf5 = os.path.join(filepath, filename)
        with h5py.File(save_to_hdf5, "w") as f:
            grp = f.create_group("weights")
            for i, w in enumerate(self.model.weights):
                grp.create_dataset(str(i), data=w.numpy())
    
    def load_weights_trajectory(self, filepath, filename='trail.h5'):
        load_from_hdf5 = os.path.join(filepath, filename)
        weights_trajectory = []
        with h5py.File(load_from_hdf5, "r") as f:
            d = f["weights"]
            for key in d.keys():
                weights_trajectory.append(tf.convert_to_tensor(d[key][:]))
        return weights_trajectory

    def _build_dataset(self, dataset_args):
        raise NotImplementedError

    def _just_build(self):
        raise NotImplementedError

    def train_step(self, x):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def self_evaluate(self):
        raise NotImplementedError

    def device_self_evaluate(self, percent=20):
        # causue uniform dataset is small, so we load them directly to gpu mem.
        pass

    def evaluate_in_all(self, inputs, labels):
        pass
