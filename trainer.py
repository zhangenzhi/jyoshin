import os
import tensorflow as tf

from models import DNN
from data_generator import read_data_from_csv
from utils import print_error


class Trainer:
    def __init__(self, args):
        self.args = args

        self._build_envs()
        self.dataset = self._build_dataset(self.args['dataset'])
        self.loss = self._build_loss(self.args['loss'])
        self.metric = self._build_metric(self.args['metric'])
        self.optimizer = self._build_optimizer(self.args['optimizer'])
        self.model = self._build_model(self.args['model'])
        
        self.just_build()

    def _build_envs(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        for item in physical_devices:
            tf.config.experimental.set_memory_growth(item, True)

    def _build_dataset(self, dataset_args):

        if dataset_args['name'] == 'uniform':
            self.x_v = None
            self.y_v = None
            path_to_data = dataset_args['path_to_data']
            dataset = read_data_from_csv(filename=path_to_data,
                                         filepath='./',
                                         batch_size=dataset_args['batch_size'],
                                         CSV_COLUMNS=['x', 'y'],
                                         num_epochs=dataset_args['epoch'])
        else:
            dataset = None

        return dataset

    def _build_model(self, model_args):
        if model_args['name'] == 'DNN':
            model = DNN(units=model_args['units'],
                        activations=model_args['activations'],
                        fuse_models=model_args['fuse_models'])
        else:
            model = None
        return model
        
    def _just_build(self):
        try:
            iter_ds = iter(self.dataset)
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'], (-1, 1))
            self.model(x['x'])
        except:
            print_error("build model with variables failed.")

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
            loss = self.loss(prediction, labels)
            grad = tape.gradient(loss, self.model.trainable_variables)

        # theta = theta - alpha * grad // optimizer
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables))

        # metric update
        self.metric.update_state(loss)

    def valid_step(self, x):
        inputs = x['x']
        prediction = self.model(inputs)

    def run(self):
        iter_ds = iter(self.dataset)

        while True:
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'], (-1, 1))
            x['y'] = tf.reshape(x['x'], (-1, 1))

            self.train_step(x)
            print("loss:", self.metric.result().numpy())
            self.metric.reset_states()

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

    def uniform_self_evaluate(self, percent=20):
        # causue uniform dataset is small, so we load them directly to gpu mem.
        iter_test = iter(self.dataset)
        self.metric.reset_states()

        all_x = []
        all_y = []
        if self.x_v == None or self.y_v == None:
            while True and percent != 0:
                try:
                    x = iter_test.get_next()
                    x['x'] = tf.reshape(x['x'], (-1, 1))
                    x['y'] = tf.reshape(x['y'], (-1, 1))
                    all_x.append(x['x'])
                    all_y.append(x['y'])
                    percent -= 1
                except:
                    print("run out of data. ")
                    break
            self.x_v = tf.concat(all_x, axis=0)
            self.y_v = tf.concat(all_y, axis=0)

        avg_loss = self.evaluate_in_all(self.x_v, self.y_v)
        avg_loss = tf.reshape(avg_loss, shape=(-1, 1))
        np_avg_loss = avg_loss.numpy()
        print("Avg loss:", np_avg_loss)
        return np_avg_loss

    # @tf.function(experimental_relax_shapes=True)
    def evaluate_in_all(self, inputs, labels):
        prediction = self.model(inputs)
        loss = self.loss(prediction, labels)
        if self.args['model']['fuse_models'] == None:
            self.metric.update_state(loss)
            avg_loss = self.metric.result()
        else:
            avg_loss = tf.reduce_mean(loss, axis=-1)

        return avg_loss

    def self_evaluate(self):
        iter_test = iter(self.dataset)
        self.metric.reset_states()

        while True:
            try:
                x = iter_test.get_next()
            except:
                print("run out of data. ")
                break
            x['x'] = tf.reshape(x['x'], (-1, 1))
            x['y'] = tf.reshape(x['y'], (-1, 1))
            prediction = self.model(x['x'])

            loss = self.loss(prediction, x['y'])
            self.metric.update_state(loss)

        avg_loss = self.metric.result().numpy()
        print("Avg loss:", avg_loss)
        return avg_loss


if __name__ == "__main__":
    trainer_args = {'loss': {'name': 'mse'},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 12, 'epoch': 3},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh']}, }

    trainer = Trainer(trainer_args)

    # trainer.save_model_weights()
    trainer.load_model_weights()
