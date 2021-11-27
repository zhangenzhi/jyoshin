import pdb

from tensorflow.python.util.compat import path_to_str
from data_generator import read_data_from_cifar10
from utils import *
from .base_trainer import BaseTrainer
import time
import tensorflow as tf
import sys
sys.path.append('..')


class Cifar10Trainer(BaseTrainer):
    def __init__(self, args):
        super(Cifar10Trainer, self).__init__(args=args)

    def _build_dataset(self, dataset_args):
        self.x_v = None
        self.y_v = None
        dataset = read_data_from_cifar10(batch_size=dataset_args['batch_size'],
                                         num_epochs=dataset_args['epoch'])
        self.plotter_dataset = read_data_from_cifar10(
            batch_size=dataset_args['batch_size'], num_epochs=1)
        return dataset

    def _just_build(self):
        try:
            iter_ds = iter(self.dataset)
            x = iter_ds.get_next()
            if self.args['dataset']['name'] == 'uniform':
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
            prediction = tf.squeeze(prediction)
            loss = self.loss(labels, prediction)
            grad = tape.gradient(loss, self.model.trainable_variables)

        # theta = theta - alpha * grad // optimizer
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables))

        # metric update
        self.metric.update_state(labels, prediction)
        return loss

    def run(self):

        iter_ds = iter(self.dataset)
        start_time = time.time()
        flag = 0
        while True:
            try:
                x = iter_ds.get_next()
            except:
                print_warning("run out of dataset.")
                break
            loss = self.train_step(x)
            if flag % 100 == 0:
                train_log = "loss:{}, metric:{}".format(
                    loss.numpy(), self.metric.result().numpy())
                print(train_log)
                write_to_file(
                    path=self.args['others']['path_to_log'], filename="train.log", s=train_log)
                self.metric.reset_states()
            flag += 1

        end_time = time.time()
        print("training cost:{}".format(end_time - start_time))

        # check if save trained model.
        if 'save_path_to_model' in self.args['model']:
            self.save_model_weights(
                filepath=self.args['model']['save_path_to_model'])

    def device_self_evaluate(self, batch_nums=10):
        # causue uniform dataset is small, so we load them directly to gpu mem.
        iter_test = iter(self.dataset)
        self.metric.reset_states()

        all_x = []
        all_y = []
        if self.x_v == None or self.y_v == None:
            while True and batch_nums != 0:
                try:
                    x = iter_test.get_next()
                    all_x.append(x['x'])
                    all_y.append(x['y'])
                    batch_nums -= 1
                except:
                    print("run out of data. ")
                    break
            self.x_v = tf.concat(all_x, axis=0)
            self.y_v = tf.concat(all_y, axis=0)

        avg_loss = self.evaluate_in_all(self.x_v, self.y_v)
        avg_loss = tf.reshape(avg_loss, shape=(-1, 1))
        np_avg_loss = avg_loss.numpy()

        return np_avg_loss

    # @tf.function(experimental_relax_shapes=True)
    def evaluate_in_all(self, inputs, labels):
        prediction = self.model(inputs)
        prediction = tf.squeeze(prediction)
        loss = self.loss(labels, prediction)
        if self.args['model']['fuse_models'] == None:
            self.metric.update_state(loss)
            avg_loss = self.metric.result()
        else:
            import pdb
            pdb.set_trace()
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
            prediction = self.model(x['x'])

            loss = self.loss(prediction, x['y'])
            self.metric.update_state(loss)

        avg_loss = self.metric.result().numpy()
        print("Avg loss:", avg_loss)
        return avg_loss


if __name__ == "__main__":
    trainer_args = {'loss': {'name': {"class_name": "SparseCategoricalCrossentropy",
                                      "config": {"from_logits": True}}},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'cifar10', 'batch_size': 32, 'epoch': 1},
                    'model': {'name': 'DNN', 'units': [64, 16, 10],
                              'activations': ['tanh', 'tanh', 'tanh'],
                              'fuse_models': 1}, }

    trainer = Cifar10Trainer(trainer_args)
    trainer.run()
