import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet.Alexnet import AlexNet
from utils.caffe_classes import class_names

IMAGENET_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

current_dir = os.getcwd()
imgs_dir = os.path.join(current_dir, 'imgs')

img_type = ['jpeg', 'png', 'bmp']
imgs_path = [os.path.join(imgs_dir, img)
             for img in os.listdir(imgs_dir)
             if img.split('.')[-1] in img_type]
assert len(imgs_path) != 0, 'Images folder is empty!!!'
imgs = [cv2.imread(img) for img in imgs_path]


# create an AlexNet object and placeholder for input
# and drapout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
rate = tf.placeholder(tf.float32)

model = AlexNet(x, rate, num_classes=1000)
score = model.fc8
softmax = tf.nn.softmax(score)

# start a Tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    fig = plt.figure(figsize=(15, 6))

    imgs_per_row = np.ceil(np.sqrt(len(imgs_path)))
    for i, img in enumerate(imgs):
        img = cv2.resize(img.astype(np.float32), (227, 227))
        img -= IMAGENET_MEAN
        img = img.reshape((1, 227, 227, 3))
        probs = sess.run(softmax, feed_dict={x: img, rate: 0})
        class_name = class_names[np.argmax(probs)]

        fig.add_subplot(imgs_per_row, imgs_per_row, 1 + i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Class: ' + class_name + ', probability: %.4f' %
                  probs[0, np.argmax(probs)])
        plt.axis('off')
    plt.show()
