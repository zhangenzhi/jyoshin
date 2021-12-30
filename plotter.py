import os
import h5py
import time
import numpy as np
import tensorflow as tf

from data_generator import read_data_from_csv
from utils import check_mkdir, print_error


class Plotter:
    def __init__(self, plotter_args, trainer):
        self.args = plotter_args
        self.step = plotter_args['step']
        self.num_evaluate = plotter_args['num_evaluate']
        self.fuse_nums = None if plotter_args['fuse_nums'] == 'None' else plotter_args['fuse_nums']
        self.trainer = trainer
        self.model = trainer.model
        self.init_weights = [tf.convert_to_tensor(
            w) for w in self.model.trainable_weights]
        self.adapt_label_dataset = self._build_adapt_label_dataset()

    def _build_adapt_label_dataset(self):
        import pdb
        pdb.set_trace()
        
        if 'localminima' in self.args.keys():
            adapt_label_dataset = self.trainer.plotter_dataset
        else:
            adapt_label_dataset = read_data_from_csv(filename="labeled.csv",
                                                    filepath=self.args["path_to_adapt_label"],
                                                    batch_size=self.trainer.args["dataset"]["batch_size"],
                                                    num_epochs=1,
                                                    CSV_COLUMNS=['y'])
        return adapt_label_dataset

    def get_init_weights(self):
        return self.init_weights

    def get_weights(self):
        return self.model.trainable_weights

    def set_weights(self, directions=None, step=0):
        # L(alpha * theta + (1- alpha)* theta') => L(theta + alpha * (theta-theta'))
        # L(theta + alpha * theta_1 + beta * theta_2)
        # Each direction have same shape with trainable weights

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            x_changes = [step[0] * d1 for d1 in dx]
            y_changes = [step[1] * d2 for d2 in dy]
            changes = [x + y for (x, y) in zip(x_changes, y_changes)]
        else:
            dx = directions[0]
            changes = [d * step for d in dx]

        # should ref in model::set fuse weight.
        init_weights = self.get_init_weights()
        trainable_variables = self.get_weights()
        for (i_w, t_w, change) in zip(init_weights, trainable_variables, changes):
            t_w.assign(i_w + change)

    def get_random_weights(self, weights):
        # random w have save shape with w
        return [tf.random.normal(w.shape) for w in weights]

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
        return normalized_direction

    def create_random_direction(self, name="x", ignore='bias_bn', norm='filter'):
        weights = self.get_weights()  # a list of parameters.

        if not self.args["load_directions"]:
            raw_direction = self.get_random_weights(weights)
            direction = self.normalize_directions_for_weights(
                raw_direction, weights, norm, ignore)
            if self.args["save_directions"]:
                self.save_directions(raw_direction, filename=name+".hdf5")
        else:
            raw_direction = self.load_directions(
                path_to_direction=self.args["save_file"], filename=name+".hdf5")
            direction = self.normalize_directions_for_weights(
                raw_direction, weights, norm, ignore)

        return direction

    def save_directions(self, directions, filename="x.hdf5"):
        save_to_hdf5 = os.path.join(self.args["save_file"], filename)
        with h5py.File(save_to_hdf5, "w") as f:
            grp = f.create_group("directions")
            for i, w in enumerate(directions):
                grp.create_dataset(str(i), data=w.numpy())

    def load_directions(self, path_to_direction, filename="x.hdf5"):
        load_from_hdf5 = os.path.join(path_to_direction, filename)
        directions = []
        with h5py.File(load_from_hdf5, "r") as f:
            d = f["directions"]
            for key in d.keys():
                directions.append(tf.convert_to_tensor(d[key][:]))
        return directions

    def plot_1d_loss(self, save_file="./result/1d", save_name='result.csv'):
        # prepare dirs
        check_mkdir(save_file)
        path_to_csv = os.path.join(save_file, save_name)

        # set init state
        directions = self.create_random_direction(
            norm='layer', name='x')

        # plot num_evaluate points in lossland
        start_time = time.time()
        for i in range(self.num_evaluate):
            step = self.step*(i-self.num_evaluate/2)
            self.set_weights(directions=[directions], step=step)
            avg_loss = self.trainer.device_self_evaluate(
                adapt_label_dataset=self.adapt_label_dataset)
            with open(path_to_csv, "ab") as f:
                np.savetxt(f, avg_loss, comments="")
        end_time = time.time()

        print("total time {}".format(end_time-start_time))

    def plot_2d_loss(self, save_file="./result/2d", save_name='result.csv'):
        # prepare dirs
        check_mkdir(save_file)
        path_to_csv = os.path.join(save_file, save_name)

        # random direction x,y
        direction_x = self.create_random_direction(
            norm='layer', name='x')
        direction_y = self.create_random_direction(
            norm='layer', name='y')
        directions = [direction_x, direction_y]

        # plot num_evaluate points in lossland
        start_time = time.time()

        for i in range(self.num_evaluate[0]):
            x_shift_step = self.step[0] * (i-self.num_evaluate[0]/2)
            for j in range(self.num_evaluate[1]):
                y_shift_step = self.step[1] * (j-self.num_evaluate[1]/2)
                step = [x_shift_step, y_shift_step]
                self.set_weights(directions=directions, step=step)
                avg_loss = self.trainer.device_self_evaluate(
                    adapt_label_dataset=self.adapt_label_dataset)
                with open(path_to_csv, "ab") as f:
                    np.savetxt(f, avg_loss, comments="")

        end_time = time.time()
        print("total time {}".format(end_time-start_time))

    def run(self):
        if self.args["task"] == "1d":
            self.plot_1d_loss(save_file=self.args['save_file'])
        elif self.args["task"] == "2d":
            self.plot_2d_loss(save_file=self.args['save_file'])
        else:
            print("No such task.")

