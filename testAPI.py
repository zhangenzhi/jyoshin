import tensorflow as tf

# y = x**2 -> y = 2x
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
    dy_dx = tape.gradient(y, x)
    print(dy_dx)

# L(x;theta) = |f(x;theta)-y| -> dL_dtheta