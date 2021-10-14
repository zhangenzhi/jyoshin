import tensorflow as tf

from models import DNN
from data_generator import read_data_from_csv


class Trainer:
    def __init__(self, args):
        self.args = args

        self.dataset = self._build_dataset(self.args['dataset'])
        self.loss = self._build_loss(self.args['loss'])
        self.metric = self._build_metric(self.args['metric'])
        self.optimizer = self._build_optimizer(self.args['optimizer'])
        self.model = self._build_model(self.args['model'])

    def _build_dataset(self, dataset_args):

        if dataset_args['name'] == 'uniform':
            dataset = read_data_from_csv(filename='labeled.csv', 
                                         filepath='./',
                                         batch_size=dataset_args['batch_size'],
                                         CSV_COLUMNS = ['x','y'],
                                         num_epochs=dataset_args['epoch'])
        else:
            dataset = None

        return dataset

    def _build_model(self, model_args):
        if model_args['name'] == 'DNN':
            model = DNN(units=model_args['units'],
                        activations=model_args['activations'])
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
            optimizer = None
        return optimizer

    def train_step(self, x):
        inputs = x['x']
        labels = x['y']

        # L(x;theta) = |f(x;theta)-y| -> dL_dtheta
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss(prediction,labels)
            grad = tape.gradient(loss, self.model.trainable_variables)
        
        # theta = theta - alpha * grad // optimizer
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        # metric update
        self.metric.update_state(loss)


    def valid_step(self):
        pass

    def run(self):
        iter_ds = iter(self.dataset)

        while True:
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'], (-1, 1))

            self.train_step(x)
            print("loss:", self.metric.result().numpy())
            self.metric.reset_states()


if __name__ == "__main__":
    trainer_args = {'loss': {'name': 'mse'},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 12, 'epoch': 3},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh']}, }

    trainer = Trainer(trainer_args)
    trainer.run()
