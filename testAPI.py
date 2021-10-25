import h5py
import tensorflow as tf
import pdb


x1 = tf.zeros(shape=(1, 64))
x2 = tf.zeros(shape=(1, 64))
x3 = tf.zeros(shape=(1, 64))

x = tf.zeros(shape=(3, 1000, 1))
b = tf.zeros(shape=(1, 64))
z = tf.stack([x1, x2, x3])
pdb.set_trace()
tf.matmul(x, z)


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
