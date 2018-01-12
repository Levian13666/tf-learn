from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.image as plimage

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

model = tf.nn.softmax(tf.matmul(x, W) + b)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


saver = tf.train.Saver()
saver.restore(sess, 'model/mnist.model')

img = plimage.imread('test.png')
img = np.fliplr(img.reshape(-1,3)).reshape(img.shape)

i = 0
image = np.multiply(np.subtract(np.reshape(img[:, :, i], [1, 784]), 1), -1)
result = sess.run(model, {x: image})
print(image)
print(np.argmax(result))

pl.imshow(img)
pl.show()
#
# image = [mnist.train.images[2535]]
# result = sess.run(model, {x: image})
# print(result)
# print(np.argmax(result))
#
# pl.imshow(np.reshape(image, [-1, 28]), cmap='Greys')
# pl.show()
