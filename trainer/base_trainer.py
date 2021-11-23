import os
import tensorflow as tf

from models import DNN
from utils import print_error

class BaseTrainer:
    def __init__(self, args):
        self.args = args

        self._build_envs()
        self.dataset = self._build_dataset(self.args['dataset'])
        self.loss = self._build_loss(self.args['loss'])
        self.metric = self._build_metric(self.args['metric'])
        self.optimizer = self._build_optimizer(self.args['optimizer'])
        self.model = self._build_model(self.args['model'])

    def _build_envs(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        for item in physical_devices:
            tf.config.experimental.set_memory_growth(item, True)

    def _build_model(self, model_args):
        if model_args['name'] == 'DNN':
            model = DNN(units=model_args['units'],
                        activations=model_args['activations'],
                        fuse_models=model_args['fuse_models'])
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
        if optimizer_args['name'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=optimizer_args['learning_rate'])
        else:
            try:
                optimizer = tf.keras.optimizers.get(optimizer_args['name'])
            except:
                print_error("no such optimizer")
        return optimizer

    def save_model_weights(self, filepath='./saved_models', name='model.h5', save_format="h5"):
        num = len(os.listdir(filepath))
        save_path = os.path.join(filepath, str(num)+'/')
        if os.path.exists(save_path):
            self.model.save_weights(save_path, save_format=save_format)
        else:
            os.mkdir(save_path)
            self.model.save_weights(save_path+name, save_format=save_format)
        print("model saved in  {}".format(save_path+name))

    def load_model_weights(self, filepath='./saved_models', num=-1, name='model.h5'):

        if num == -1:
            # -1 means latest model
            num = len(os.listdir(filepath)) - 1
        filepath = os.path.join(filepath, str(num)+'/')

        if os.path.exists(filepath):
            self.just_build()
            self.model.load_weights(filepath+name)
            print("model load from {}".format(filepath+name))
        else:
            print("path doesn't exits.")

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

    # @tf.function(experimental_relax_shapes=True)
    def evaluate_in_all(self, inputs, labels):
        pass
