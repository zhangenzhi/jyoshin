import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import random

from trainer import Trainer


class Plotter:
    def __init__(self, plotter_args, model):
        self.step = plotter_args['step']
        self.num_evaluate = plotter_args['num_evaluate']
        self.fuse_models = plotter_args['fuse_models']
        self.model = model

    def get_weights(self):
        return self.model.trainable_weights

    def fuse_directions(self, normalized_directions):
        random_directions = []
        for d in normalized_directions:
            fuse_random_direction = []
            for i in range(self.fuse_models):
                fuse_random_direction.append(d * (i+1))
            random_directions.append(tf.stack(fuse_random_direction))
        return random_directions

    def set_weights(self, directions=None, init_state=False):
        # L(alpha * theta + (1- alpha)* theta') => L(theta + alpha * (theta-theta'))
        # L(theta + alpha * theta_1 + beta * theta_2)
        # Each direction have same shape with trainable weights
        if directions == None:
            print("None of directions.")
        elif init_state == True:
            if len(directions) == 2:
                pass
            else:
                shift = -self.step*self.fuse_models/2
                changes = [d*shift for d in directions[0]]
        else:
            if self.fuse_models == None:
                if len(directions) == 2:
                    dx = directions[0]
                    dy = directions[1]
                    changes = [self.step[0]*d0 + self.step[1] *
                               d1 for (d0, d1) in zip(dx, dy)]
                else:
                    changes = [d*self.step for d in directions[0]]
            else:
                if len(directions) == 2:
                    pass
                else:
                    changes = [d*self.step for d in directions[0]]

        weights = self.get_weights()
        for (weight, change) in zip(weights, changes):
            weight.assign_add(change)

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
            normalized_direction = direction * tf.norm(weights) / tf.norm(direction)
        elif norm == 'weight':
            normalized_direction = direction * weights
        elif norm == 'd_filter':
            normalized_direction = []
            for d in direction:
                normalized_direction.append(d / (tf.norm(d)+1e-10))
        elif norm == 'd_layer':
            normalized_direction = direction / tf.norm(direction)

        if self.fuse_models != None:
            normalized_direction = self.fuse_directions(normalized_direction)
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
        return normalized_direction

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

    def cal_loss_surf(self, surf_file):
        pass


if __name__ == '__main__':
    trainer_args = {'loss': {'name': 'mse'},
                    'metric': {'name': 'Mean'},
                    'optimizer': {'name': 'SGD', 'learning_rate': 0.001},
                    'dataset': {'name': 'uniform', 'batch_size': 100, 'epoch': 1},
                    'model': {'name': 'DNN', 'units': [64, 16, 1],
                              'activations': ['tanh', 'tanh', 'tanh']}, }

    trainer = Trainer(trainer_args)
    trainer.just_build()
    trainer.model.summary()
    # trainer.self_evaluate()

    plotter = Plotter(trainer.model)
    normalized_random_direction = plotter.create_random_direction(norm='layer')

    N = 1000
    step = 1/100
    # set init state
    plotter.set_weights([normalized_random_direction], step=-step*N/2)

    # plot N points in lossland

    for i in range(N):
        plotter.set_weights([normalized_random_direction], step=step)
        avg_loss = trainer.self_evaluate()
        with open("result_1000.csv", "ab") as f:
            np.savetxt(f, [avg_loss], comments="")
