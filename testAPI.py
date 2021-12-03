import pdb
import tensorflow as tf
import h5py
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

physical_devices = tf.config.list_physical_devices('GPU')
for item in physical_devices:
    tf.config.experimental.set_memory_growth(item, True)

i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: (tf.add(i, 1), )
r = tf.while_loop(c, b, [i])

# with tf.device("/device:gpu:0"):
#     # data = tf.zeros(shape=[50000, 32, 32, 3])
#     slice_y = tf.zeros(shape=[32*32*3, 1])
#     slice_data = tf.zeros(shape=[500, 32, 32, 3])
#     slice_data = tf.reshape(slice_data, shape=(500, -1))

#     # while True:
#     #     # for i in range(100):
#     #     output = tf.matmul(slice_data, slice_y)
#     x = slice_data
#     y = slice_y
#     i = tf.constant(0)
#     def f(i, x, y): return tf.matmul(x, y)
#     def c(i, x, y): return tf.less(i, 2 ** 20)
#     r = tf.while_loop(cond=c, body=f, loop_vars=(i, x, y))

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
