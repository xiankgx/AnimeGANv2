import tensorflow as tf
import tensorflow.contrib as tf_contrib

from .ops import Conv2D, Conv2DNormLReLU, InvertedRes_block, Upsample


class G_net(object):

    def __init__(self, inputs):
        with tf.variable_scope('G_MODEL'):
            with tf.variable_scope('A'):
                inputs = Conv2DNormLReLU(inputs, 32, 7)
                inputs = Conv2DNormLReLU(inputs, 64, strides=2)
                inputs = Conv2DNormLReLU(inputs, 64)

            with tf.variable_scope('B'):
                inputs = Conv2DNormLReLU(inputs, 128, strides=2)
                inputs = Conv2DNormLReLU(inputs, 128)

            with tf.variable_scope('C'):
                inputs = Conv2DNormLReLU(inputs, 128)
                inputs = InvertedRes_block(inputs, 2, 256, 1, 'r1')
                inputs = InvertedRes_block(inputs, 2, 256, 1, 'r2')
                inputs = InvertedRes_block(inputs, 2, 256, 1, 'r3')
                inputs = InvertedRes_block(inputs, 2, 256, 1, 'r4')
                inputs = Conv2DNormLReLU(inputs, 128)

            with tf.variable_scope('D'):
                inputs = Upsample(inputs, 128)
                inputs = Conv2DNormLReLU(inputs, 128)

            with tf.variable_scope('E'):
                inputs = Upsample(inputs, 64)
                inputs = Conv2DNormLReLU(inputs, 64)
                inputs = Conv2DNormLReLU(inputs, 32, 7)

            with tf.variable_scope('out_layer'):
                out = Conv2D(inputs, filters=3, kernel_size=1, strides=1)
                self.fake = tf.tanh(out)
