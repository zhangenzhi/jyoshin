import h5py
import tensorflow as tf
import pdb

with tf.device("/device:gpu:0"):
    data = tf.zeros(shape=[50000, 32, 32, 3])
    slice_y = tf.zeros(shape=[32*32*3,1])
    while True:
        for i in range(100):
            # slice_data = data[500*i:500*(i+1)]
            slice_data = tf.reshape(data[500*i:500*(i+1)], shape=(500, -1))
            output = slice_data * slice_y

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
