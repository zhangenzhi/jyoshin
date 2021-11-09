import sys
sys.path.append('..')

from utils import print_error
from data_generator import read_data_from_csv
from .base_trainer import BaseTrainer
import tensorflow as tf


class UniformTrainer(BaseTrainer):
    def __init__(self, args):
        super(UniformTrainer, self).__init__(args=args)

    def _build_dataset(self, dataset_args):
        self.x_v = None
        self.y_v = None
        dataset = read_data_from_csv(filename=dataset_args['path_to_data'],
                                     batch_size=dataset_args['batch_size'],
                                     CSV_COLUMNS=['x','y'],
                                     num_epochs=dataset_args['epoch'])
        return dataset

    def _just_build(self):
        try:
            iter_ds = iter(self.dataset)
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'], (-1, 1))
            self.model(x['x'])
        except:
            print_error("build model with variables failed.")

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

    def run(self):
        iter_ds = iter(self.dataset)

        while True:
            x = iter_ds.get_next()
            x['x'] = tf.reshape(x['x'], (-1, 1))
            x['y'] = tf.reshape(x['x'], (-1, 1))

            self.train_step(x)
            print("loss:", self.metric.result().numpy())
            self.metric.reset_states()

    def device_self_evaluate(self, percent=20):
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
        # print("Avg loss:", np_avg_loss)
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
    trainer_args = {'loss': {'name': "mse"},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 32, 'epoch': 1},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh'],
                              'fuse_models': 1}, }

    trainer = UniformTrainer(trainer_args)
    trainer.run()
