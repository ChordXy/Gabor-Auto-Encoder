'''
@Author: Cabrite
@Date: 2019-11-04 19:45:33
@LastEditTime: 2019-11-04 22:00:51
@LastEditors: Please set LastEditors
@Description: AE
@FilePath: \Gabor-Auto-Encoder\library\My Auto-Encoder.py
'''
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pylab

def VisulizeImage(src_images, cor_images, res_images):
    fig, ax = plt.subplots(nrows=3, ncols=6, sharex='all', sharey='all')
    ax = ax.flatten()
    all_images = []
    for image in src_images:
        all_images.append(image)
    for image in cor_images:
        all_images.append(image)
    for image in res_images:
        all_images.append(image)

    for i, image in enumerate(all_images):
        img = image.reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    
######################## Defining Parameters ########################
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = int(mnist.train.num_examples)
training_epochs = 30
batch_size = 128
total_batch = int(n_samples / batch_size)
display_step = 1
test_data = mnist.test.images[0:6]

n_inputs = 784
n_hidden = 50
n_outputs = 784

######################## Defining Network ########################
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])
keep_prob = tf.placeholder(tf.float32)

# Add noise
corruption = tf.nn.dropout(x, keep_prob)

Encoder_Weight = tf.Variable(tf.random_normal([n_inputs, n_hidden]))
Encoder_bias = tf.Variable(tf.zeros([n_hidden]))
Encoder_output = tf.nn.softmax(tf.matmul(corruption, Encoder_Weight) + Encoder_bias)

Decoder_Weight = tf.Variable(tf.random_normal([n_hidden, n_outputs]))
Decoder_bias = tf.Variable(tf.zeros([n_outputs]))
Decoder_output = tf.nn.softmax(tf.matmul(Encoder_output, Decoder_Weight) + Decoder_bias)

cost = tf.reduce_sum(tf.pow(Decoder_output - x, 2))
learning_rate = 0.1
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

######################## Training Network ########################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, batch_cost = sess.run([optimizer, cost], feed_dict={x:batch_xs, keep_prob:0.5})
            avg_cost += batch_cost / total_batch
        if (epoch + 1) % display_step == 0:
            print("Epoch {:3}: cost = {:.9f}".format(epoch + 1, avg_cost))

    cor = corruption.eval(feed_dict={x:test_data, keep_prob:0.5})
    out = Decoder_output.eval(feed_dict={x:test_data, keep_prob:0.5})
    VisulizeImage(test_data, cor[0:6], out[0:6])
    print("Finished!")

######################## Testing Network ########################
