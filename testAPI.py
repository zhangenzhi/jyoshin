import tensorflow as tf
import pdb
import h5py
import time
import os

from tensorflow.python.keras.layers.core import Lambda
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


physical_devices = tf.config.list_physical_devices('GPU')
for item in physical_devices:
    tf.config.experimental.set_memory_growth(item, True)


def on_device_matmul():
    with tf.device("/device:gpu:0"):
        slice_y = tf.zeros(shape=[32*32*3, 1])
        slice_data = tf.zeros(shape=[500, 32, 32, 3])
        slice_data = tf.reshape(slice_data, shape=(500, -1))

        # for-loop: total time: 118.20909833908081
        # for i in tf.range(1, 2**20):
        #     output = tf.matmul(slice_data, slice_y)

        # whille-loop: total time: 49.9278666973114
        # 8-body:  total time: 32.072572231292725
        # 16-body: total time: 30.435903072357178
        # 32-body: total time: 28.816709756851196
        # i = tf.constant(0)
        # while tf.less(i, 2**20):
        #     output = tf.matmul(slice_data, slice_y)
        #     i = tf.add(i, 1)

        # tf.while_loop
        i = tf.constant(0)
        def c(i): 
            return tf.less(i, 2**20)
        def f(i, x, y):
            output = tf.matmul(x, y)
            return i + 1, x, y
        r = tf.while_loop(cond=c, body=f, loop_vars=[i, slice_data, slice_y])


if __name__ == '__main__':
    start = time.time()
    # on_device_matmul()
    end = time.time()
    print("total time: {}".format(end-start))

    # x = slice_data
    # y = slice_y
    # i = tf.constant(0)
    # def f(i, x, y): return tf.matmul(x, y)
    # def c(i, x, y): return tf.less(i, 2 ** 20)
    # r = tf.while_loop(cond=c, body=f, loop_vars=(i, x, y))

# f1 = h5py.File("./saved_models/1/model.h5")
# f2 = h5py.File("./saved_models/2/model.h5")
# f1.keys


# y = x**2 -> y = 2x
# x = tf.Variable(3.0)
# with tf.GradientTape() as tape:
#     y = x**2
#     dy_dx = tape.gradient(y, x)
#     print(dy_dx)

# L(x;theta) = |f(x;theta)-y| -> dL_dtheta
