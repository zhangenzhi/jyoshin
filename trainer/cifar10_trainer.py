from tensorflow.python.util.compat import path_to_str
from data_generator import read_data_from_cifar10, read_data_from_csv
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

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x):
        inputs = x['x']
        labels = x['y']

        # L(x;theta) = |f(x;theta)-y| -> dL_dtheta
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss(labels, prediction)
            grad = tape.gradient(loss, self.model.trainable_variables)

        # theta = theta - alpha * grad // optimizer
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables))

        # metric update
        self.metric.update_state(labels, prediction)
        return loss

    @tf.function(experimental_relax_shapes=True)
    def distribute_train_step(self, x):
        per_replica_losses = self.strategy.run(self.train_step, args=(x,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=0)

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
            # train step
            if self.args['others']['distribute']:
                loss = self.distribute_train_step(x)
            else: 
                loss = self.train_step(x)
                
            if flag % 100 == 0:
                train_log = "step:{},loss:{}, metric:{}".format(flag,
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

    def device_self_evaluate(self, adapt_label_dataset, batch_nums=2):
        # causue cifar10 dataset is small, so we load them directly to gpu mem.

        if self.x_v == None or self.y_v == None:
            all_x = []
            all_y = []
            iter_test = iter(self.plotter_dataset)
            iter_label = iter(adapt_label_dataset)
            for _ in range(batch_nums):
                try:
                    x = iter_test.get_next()
                    y = iter_label.get_next()
                    all_x.append(x['x'])
                    all_y.append(y['y'])
                    batch_nums -= 1
                except:
                    print_error("run out of data to put in device. ")
                    break
            with tf.device("/device:gpu:0"):
                self.x_v = tf.concat(all_x, axis=0)
                self.y_v = tf.concat(all_y, axis=0)

        with tf.device("/device:gpu:0"):
            _, avg_metric = self.evaluate_in_all(self.x_v, self.y_v)
            avg_metric = tf.constant(1.0) - self.metric.result()
            avg_metric = tf.reshape(avg_metric, shape=(-1, 1))
        np_avg_metric = avg_metric.numpy()

        return np_avg_metric

    # @tf.function(experimental_relax_shapes=True)
    def evaluate_in_all(self, inputs, labels):
        prediction = self.model(inputs)

        # loss
        loss = self.loss(labels, prediction)

        # metric
        self.metric.reset_states()
        metric = self.metric.update_state(labels, prediction)

        return loss, metric


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
