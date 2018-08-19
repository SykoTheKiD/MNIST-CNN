import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

FLAGS = tf.app.flags.FLAGS
SPEC_SENTINEL = "*"

def init_flags():
    tf.app.flags.DEFINE_integer('input_dim', 28, "Image Dimension")
    tf.app.flags.DEFINE_integer('output_dim', 10, "Image Dimension")
    tf.app.flags.DEFINE_integer('num_epochs', 10, "Number of epochs")
    tf.app.flags.DEFINE_integer('batch_size', 128, "Number of samples per batch.")
    tf.app.flags.DEFINE_integer('nb_batch_per_epoch', 100, "Number of batches per epoch")
    tf.app.flags.DEFINE_float('learning_rate', 1E-4, "Learning rate used for AdamOptimizer")

    tf.app.flags.DEFINE_string('model_dir', './models', "Output folder where checkpoints are dumped.")
    tf.app.flags.DEFINE_string('log_dir', './logs', "Logs for tensorboard.")

class NetworkBuilder:

    def add_conv_layer(self, input_layer, output_size=32, feature_size=(5, 5), strides=(1, 1, 1, 1), padding='SAME', summary=False):
        with tf.name_scope("Convolution"):
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([feature_size[0], feature_size[1], input_size, output_size]), name='conv_weights')
            biases = tf.Variable(tf.random_normal([output_size]), name='conv_biases')
            return tf.add(tf.nn.conv2d(input_layer, weights, strides=strides, padding=padding), biases)

    def add_pooling_layer(self, input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
        with tf.name_scope("Pooling"):
            return tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding=padding)

    def add_relu_layer(self, input_layer):
        with tf.name_scope("Activation"):
            return tf.nn.relu(input_layer)
    
    def flatten(self, input_layer):
        with tf.name_scope("Flatten"):
            input_size = input_layer.get_shape().as_list()
            new_size = input_size[-1] * input_size[-2] * input_size[-3]
            return tf.reshape(input_layer, [-1, new_size])
    
    def add_softmax_layer(self, input_layer):
        with tf.name_scope("Activation"):
            return tf.nn.softmax(input_layer)

    def add_fully_connected_layer(self, input_layer, size, summary=False):
        with tf.name_scope("Fully_Connected"):
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([input_size, size]), name="dense_weight")
        biases = tf.Variable(tf.random_normal([size]), name="fully_connected_biases")
        fc = tf.add(tf.matmul(input_layer, weights), biases)
        return fc

class Dataset:
    def __init__(self, csv_path, spec, test_size=0.2):
        frame = pd.read_csv(csv_path)
        self.train, self.validation = train_test_split(frame, test_size=test_size)
        for col in frame:
            col_func = spec[col] if col in spec else spec[SPEC_SENTINEL]
            col.apply(lambda x: col_func(x))

        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []



class CNNModel:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=[None, FLAGS.input_dim, FLAGS.img.dim], name="input_image")
        self.output = tf.placeholder(tf.int32, shape=[None, FLAGS.output_dim], name="output")
        nb = NetworkBuilder(self.input)
        out_layer = nb.add_conv_layer(self.input, 14)
        out_layer = nb.add_pooling_layer(out_layer)
        out_layer = nb.flatten(out_layer)
        self.out_layer = nb.add_fully_connected_layer(out_layer, FLAGS.output_dim)
        self.prediction = nb.add_softmax_layer(out_layer)

        with tf.name_scope("Optimization"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.out_layer, labels=self.output)
            self.cost = tf.reduce_mean(cost)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

        with tf.name_scope("Accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            for epoch in range(FLAGS.num_epochs):
                loss = 0
                current_batch = 0
                while current_batch < len(train_x):
                    start = 1
                    end = current_batch + FLAGS.batch_size
                    batch_x = np.array(train_x[start: end])
                    batch_y = np.array(train_y[start: end])
                    epoch_cost, epoch_accuracy, steps, _ = sess.run([self.cost, self.accuracy, self.global_step, self.optimizer], feed_dict={self.input: batch_x, self.output: batch_y})
                    loss += epoch_cost
                    if steps % 100 == 0:
                        saver.save(sess, FLAGS.model_dir, global_step=steps)
                print("Epoch {0}: completed out of {1} | loss: {2} | accuracy: {3}".format(epoch, FLAGS.num_epochs, epoch_cost, epoch_accuracy))

            
            correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print("Accuracy:", accuracy.eval({self.input: test_x, self.output: test_y}))

    def predict(self):
        pass

def main():
    # read file
    init_flags()
    dataset_spec = {
        "label": tf.int32,
        "*": tf.int32
    }
    dset = Dataset("./data/train.csv", dataset_spec)
    cnn = CNNModel()
    cnn.train(dset.train_x, dset.train_y, dset.test_x, dset.test_y)
    cnn.predict()

if __name__ == '__main__':
    main()