import tensorflow as tf
import pandas as pd


class VGG16():

    def __init__(self, inputs, rate, num_classes):
        self.In_Data = inputs
        self.rate = rate
        self.num_classes = num_classes
        self.receptive_field = {'layer\'s name': ['0th layer'],
                                'receptive_field': [1], 'product_Si': [1]}
        self._create_network()

    def _create_network(self):
        # 1st layer: conv3-64(W, Relu)
        conv1, dicts = conv(self.In_Data, filter_size=3, num_filters=64,
                            stride=1, padding='VALID', name='conv1',
                            dicts=self.receptive_field)
        # 2nd layer: conv3-64(W, Relu) --> max pooling
        conv2, dicts = conv(conv1, filter_size=3, num_filters=64,
                            stride=1, padding='VALID', name='conv2',
                            dicts=dicts)
        pool2, dicts = max_pool(conv2, filter_size=2, stride=2,
                                padding='VALID', name='pool2',
                                dicts=dicts)
        # 3rd layer: conv3-128(W, Relu)
        conv3, dicts = conv(pool2, filter_size=3, num_filters=128,
                            stride=1, padding='VALID', name='conv3',
                            dicts=dicts)
        # 4th layer: conv3-128(W, Relu) --> max pooling
        conv4, dicts = conv(conv3, filter_size=3, num_filters=128,
                            stride=1, padding='VALID', name='conv4',
                            dicts=dicts)
        pool4, dicts = max_pool(conv4, filter_size=2, stride=2,
                                padding='VALID', name='pool4',
                                dicts=dicts)
        # 5th layer: conv3-256(W, Relu)
        conv5, dicts = conv(pool4, filter_size=3, num_filters=256,
                            stride=1, padding='VALID', name='conv5',
                            dicts=dicts)
        # 6th layer: conv3-256(W, Relu)
        conv6, dicts = conv(conv5, filter_size=3, num_filters=256,
                            stride=1, padding='VALID', name='conv6',
                            dicts=dicts)
        # 7th layer: conv3-256(W, Relu) --> max pooling
        conv7, dicts = conv(conv6, filter_size=3, num_filters=256,
                            stride=1, padding='VALID', name='conv7',
                            dicts=dicts)
        pool7, dicts = max_pool(conv7, filter_size=2, stride=2,
                                padding='VALID', name='pool7',
                                dicts=dicts)
        # 8th layer: conv3-512(W, Relu)
        conv8, dicts = conv(pool7, filter_size=3, num_filters=512,
                            stride=1, padding='VALID', name='conv8',
                            dicts=dicts)
        # 9th layer: conv3-512(W, Relu)
        conv9, dicts = conv(conv8, filter_size=3, num_filters=512,
                            stride=1, padding='VALID', name='conv9',
                            dicts=dicts)
        # 10th layer: conv3-512(W, Relu) --> max pooling
        conv10, dicts = conv(conv9, filter_size=3, num_filters=512,
                             stride=1, padding='VALID', name='conv10',
                             dicts=dicts)
        pool10, dicts = max_pool(conv10, filter_size=2, stride=2,
                                 padding='VALID', name='pool10',
                                 dicts=dicts)
        # 11th layer: conv3-512(W, Relu)
        conv11, dicts = conv(pool10, filter_size=3, num_filters=512,
                             stride=1, padding='VALID', name='conv11',
                             dicts=dicts)
        # 12th layer: conv3-512(W, Relu)
        conv12, dicts = conv(conv11, filter_size=3, num_filters=512,
                             stride=1, padding='VALID', name='conv12',
                             dicts=dicts)
        # 13th layer: conv3-512(W, Relu) --> max pooling
        conv13, dicts = conv(conv12, filter_size=3, num_filters=512,
                             stride=1, padding='VALID', name='conv13',
                             dicts=dicts)
        pool13, dicts = max_pool(conv13, filter_size=2, stride=2,
                                 padding='VALID', name='pool13',
                                 dicts=dicts)
        self.receptive_field = dicts
        # 14th layer: FC-4096(W, Relu) --> dropout
        flattened = tf.reshape(pool13, [-1, 1 * 1 * 512])
        fc14 = fc(flattened, input_size=512, output_size=4096, name='fc14')
        dropout14 = dropout(fc14, rate=self.rate, name='dropout14')
        # 15th layer: FC-4096(W, Relu) --> dropout
        fc15 = fc(dropout14, input_size=4096, output_size=4096, name='fc15')
        dropout15 = dropout(fc15, rate=self.rate, name='dropout15')
        # 16th layer: FC-1000(W softmax) --> output
        self.output = fc(dropout15, input_size=4096,
                         output_size=self.num_classes, relu=False, name='fc16')


def conv(input_data, filter_size, num_filters, stride, padding, name, dicts):

    input_channels = int(input_data.shape[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(
            'weights', shape=[filter_size, filter_size,
                              input_channels, num_filters])
        biases = tf.get_variable('biases', [num_filters])
        conv = tf.nn.conv2d(input_data, weights, strides=[
                            1, stride, stride, 1], padding=padding)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        dicts['receptive_field'].append(dicts['receptive_field'][-1] +
                                        (filter_size - 1) *
                                        dicts['product_Si'][-1])
        dicts['product_Si'].append(dicts['product_Si'][-1] * stride)
        dicts['layer\'s name'].append(name)

        return tf.nn.relu(bias, name=scope.name), dicts


def max_pool(input_data, filter_size, stride, padding, name, dicts):

    dicts['receptive_field'].append(dicts['receptive_field'][-1] +
                                    (filter_size - 1) *
                                    dicts['product_Si'][-1])
    dicts['product_Si'].append(dicts['product_Si'][-1] * stride)
    dicts['layer\'s name'].append(name)

    return tf.nn.max_pool(input_data, ksize=[1, filter_size, filter_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding, name=name), dicts


def fc(input_data, input_size, output_size, name, relu=True):

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[input_size, output_size])
        biases = tf.get_variable('biases', shape=[output_size])
        logits = tf.nn.xw_plus_b(input_data, weights, biases, name=scope.name)

        if relu:
            output = tf.nn.relu(logits)
        else:
            output = tf.nn.softmax(logits, axis=1, name='softmax')

        return output


def dropout(input_data, rate, name):

    return tf.nn.dropout(input_data, rate, name=name)


if __name__ == '__main__':
    batch_size = 64
    num_classes = 1000
    X = tf.placeholder(tf.float32, shape=[
                       batch_size, 224, 224, 3], name='input')
    rate = tf.placeholder(tf.float32, name='rate')
    model = VGG16(X, rate, num_classes)
    df = pd.DataFrame(model.receptive_field)
    df.to_csv('./tensorboard/receptive_field.csv', index=None)
    with tf.Session() as sess:
        tf.summary.FileWriter('./tensorboard', sess.graph)
