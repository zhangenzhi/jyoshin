import h5py
import tensorflow as tf
import pdb


x1 = tf.zeros(shape=(100,10))
x2 = tf.zeros(shape=(100,10))
x3 = tf.zeros(shape=(100,10))
pdb.set_trace()
z = tf.stack([x1,x2,x3])


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