import tensorflow as tf

from models import DNN
from data_generator import read_data_from_csv


class Trainer:
    def __init__(self, args):
        self.args = args

        self.dataset = self._build_dataset(self.args['dataset'])
        self.loss = self._build_loss(self.args['loss'])
        self.optimizer = self._build_optimizer(self.args['optimizer'])
        self.model = self._build_model(self.args['model'])

    def _build_dataset(self, dataset_args):

        if dataset_args['name'] == 'uniform':
            dataset = read_data_from_csv(filename='uniform.csv', filepath='./')
        else:
            dataset = None

        dataset = dataset.batch(dataset_args['batch_size'])
        dataset = dataset.repeat(dataset_args['epoch'])
        return dataset

    def _build_model(self, model_args):
        if model_args['name'] == 'uniform':
            model = DNN(units=model_args['units'],
                        activations=model_args['activations'])
        else:
            model = None
        return model

    def _build_loss(self, loss_args):
        loss = tf.keras.losses.get(loss_args['name'])
        return loss

    def _build_optimizer(self, optimizer_args):
        if optimizer_args['name'] == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=optimizer_args['learning_rate'])
        else:
            optimizer = None
        return optimizer

    def train_step(self, x):
        pass

    def valid_step(self):
        pass

    def run(self):
        iter_ds = iter(self.dataset)

        while True:
            x = iter_ds.get_next()
            self.train_step(x)


if __name__ == "__main__":
    trainer_args = {'loss': {'name': 'mse'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 12, 'epoch': 3},
                    'model': {'name': 'dnn', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh']}, }

    trainer = Trainer(trainer_args)
