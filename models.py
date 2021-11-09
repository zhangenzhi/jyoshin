from os import name
import tensorflow as tf
from tensorflow import keras

from data_generator import read_data_from_csv


class Linear(keras.layers.Layer):
    def __init__(self, units=32, fuse_layers=None):
        super(Linear, self).__init__()
        self.units = units
        self.fuse_layers = fuse_layers

    def build(self, input_shape):

        if self.fuse_layers == None:
            w_init = tf.random_normal_initializer(seed=100000)
            b_init = tf.zeros_initializer()

            self.w = tf.Variable(
                initial_value=w_init(
                    shape=(input_shape[-1], self.units), dtype="float32"),
                trainable=True, name="w"
            )
            self.b = tf.Variable(
                initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=True,
                name="b"
            )
        else:
            w_init = tf.random_normal_initializer(seed=100000)
            b_init = tf.zeros_initializer()

            init_w = tf.stack(
                [w_init(shape=(input_shape[-1], self.units), dtype="float32")] * self.fuse_layers)
            init_b = tf.stack([b_init(
                shape=(1, self.units), dtype="float32")] * self.fuse_layers)

            self.fuse_w = tf.Variable(
                initial_value=init_w, trainable=True, name="w")
            self.fuse_b = tf.Variable(
                initial_value=init_b, trainable=True,
                name="b"
            )

    def call(self, inputs):
        if self.fuse_layers == None:
            outputs = tf.matmul(inputs, self.w) + self.b
        else:
            outputs = tf.matmul(inputs, self.fuse_w) + self.fuse_b
        return outputs


class DNN(tf.keras.Model):
    def __init__(self,
                 units=[64, 32, 16, 1],
                 activations=['tanh', 'tanh', 'tanh', 'tanh'],
                 fuse_models=None):
        super(DNN, self).__init__()

        self.units = units
        self.activations = activations
        self.fuse_models = fuse_models
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layers = self._build_fc()
        self.fc_act = self._build_act()

    def _build_fc(self):
        layers = []
        for units in self.units:
            layers.append(Linear(units=units, fuse_layers=self.fuse_models))
        return layers

    def _build_act(self):
        acts = []
        for act in self.activations:
            acts.append(tf.keras.layers.Activation(act))
        return acts

    def call(self, inputs):
        x = inputs
        x = self.flatten(x)
        for layer, act in zip(self.fc_layers, self.fc_act):
            x = layer(x)
            x = act(x)
        return x


if __name__ == "__main__":
    dataset = read_data_from_csv()
    iter_ds = iter(dataset)
    x = iter_ds.get_next()
    # l1 = Linear()
    x['x'] = tf.reshape(x['x'], (-1, 1))
    # output = l1(x['x'])

    dnn = DNN()
    output = dnn(x['x'])
    print(output)
