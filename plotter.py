import os
import h5py
import time
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import random
from tensorflow.python.ops.variables import trainable_variables

from trainer import Trainer


class Plotter:
    def __init__(self, plotter_args, model):
        self.step = plotter_args['step']
        self.num_evaluate = plotter_args['num_evaluate']
        self.fuse_models = plotter_args['fuse_models']
        self.model = model
        self.init_weights = [tf.convert_to_tensor(
            w) for w in self.model.trainable_weights]

    def get_init_weights(self):
        return self.init_weights

    def get_weights(self):
        return self.model.trainable_weights

    def fuse_directions(self, normalized_directions):
        shift_directions = []
        base_directions = []
        for d in normalized_directions:
            shift_direction = []
            base_direction = []
            for i in range(self.fuse_models):
                base_direction.append(d)
                shift_direction.append(d * (i*1))
            shift_directions.append(tf.stack(shift_direction))
            base_directions.append(tf.stack(base_direction))
        return base_directions, shift_directions

    def set_weights(self, directions=None, step=0):
        # L(alpha * theta + (1- alpha)* theta') => L(theta + alpha * (theta-theta'))
        # L(theta + alpha * theta_1 + beta * theta_2)
        # Each direction have same shape with trainable weights

        if len(directions) == 2:
            # just fuse models in y direction.
            dx = directions[0]
            dy = directions[1]
            changes = [step[0] * d0 + self.step[1] * d2 + step[1] * d1
                       for (d0, d1, d2) in zip(dx[0], dy[0], dy[1])]
        else:
            dx = directions[0]
            changes = [d * step *
                       self.fuse_models for d in dx]

        init_weights = self.get_init_weights()
        trainable_variables = self.get_weights()
        for (i_w, w, change) in zip(init_weights, trainable_variables, changes):
            w.assign(i_w + change)

    def get_random_weights(self, weights):
        # random w have save shape with w
        if self.fuse_models == None:
            return [tf.random.normal(w.shape) for w in weights]
        else:
            single_random_direction = []
            for w in weights:
                dims = list(w.shape)
                single_random_direction.append(
                    tf.random.normal(shape=dims[1:]))
            return single_random_direction

    def get_diff_weights(self, weights_1, weights_2):
        return [w2 - w1 for (w1, w2) in zip(weights_1, weights_2)]

    def normalize_direction(self, direction, weights, norm='filter'):

        if norm == 'filter':
            # filter normalize: d = direction / norm(direction) * norm(weight)
            normalized_direction = []
            for d, w in zip(direction, weights):
                normalized_direction.append(
                    d / (tf.norm(d) + 1e-10) * tf.norm(w))
        elif norm == 'layer':
            # filter normalize: normalzied_d = direction / norm(direction) * norm(weight)
            normalized_direction = direction * \
                tf.norm(weights) / tf.norm(direction)
        elif norm == 'weight':
            normalized_direction = direction * weights
        elif norm == 'd_filter':
            normalized_direction = []
            for d in direction:
                normalized_direction.append(d / (tf.norm(d)+1e-10))
        elif norm == 'd_layer':
            normalized_direction = direction / tf.norm(direction)
        return normalized_direction

    def normalize_directions_for_weights(self, direction, weights, norm="filter", ignore="bias_bn"):
        normalized_direction = []
        for d, w in zip(direction, weights):
            if len(d.shape) <= 1:
                if ignore == "bias_bn":
                    d = tf.zeros(d.shape)
                else:
                    d = w
                normalized_direction.append(d)
            else:
                normalized_direction.append(
                    self.normalize_direction(d, w, norm))
        fused_normalized_direction = []
        if self.fuse_models != None:
            fused_normalized_direction = self.fuse_directions(
                normalized_direction)
        return fused_normalized_direction

    def create_target_direction(self):
        pass

    def create_random_direction(self, ignore='bias_bn', norm='filter'):
        weights = self.get_weights()  # a list of parameters.
        direction = self.get_random_weights(weights)
        direction = self.normalize_directions_for_weights(
            direction, weights, norm, ignore)
        return direction

    def setup_direction(self):
        pass

    def name_direction_file(self):
        pass

    def load_directions(self):
        pass

    def plot_1d_loss(self, trainer, save_csv="./result.csv"):
        # set init state
        fused_direction, _ = self.create_random_direction(
            norm='layer')
        directions = fused_direction

        # plot num_evaluate * fuse_models points in lossland
        start_time = time.time()
        for i in range(self.num_evaluate):
            step = self.step*(i-self.num_evaluate/2)
            self.set_weights(directions=[directions], step=step)
            avg_loss = trainer.uniform_self_evaluate()
            with open(save_csv, "ab") as f:
                np.savetxt(f, avg_loss, comments="")
        end_time = time.time()

        print("total time {}".format(end_time-start_time))

    def plot_2d_loss(self, trainer, save_csv="./result.csv"):
        # random direction x,y
        direction_x = self.create_random_direction(
            norm='layer')
        direction_y = self.create_random_direction(
            norm='layer')
        directions = [direction_x, direction_y]

        # plot num_evaluate * fuse_models points in lossland
        start_time = time.time()

        for i in range(self.num_evaluate[0]):
            for j in range(self.num_evaluate[1]):
                x_shift_step = self.step[0]*(i-self.num_evaluate[0]/2)
                y_shift_step = self.step[1]*(j-self.num_evaluate[1]/2) * self.fuse_models
                step = [x_shift_step, y_shift_step]
                self.set_weights(directions=directions, step=step)
                avg_loss = trainer.uniform_self_evaluate()
                with open(save_csv, "ab") as f:
                    np.savetxt(f, avg_loss, comments="")

        end_time = time.time()
        print("total time {}".format(end_time-start_time))
