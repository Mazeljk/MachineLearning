"""
Realize AI-player in Doom's basic scene(only three actions:left, right, shot) with deep Q-learning
dependencies: python 3.7.1, vizdoom 1.1.7
"""
import tensorflow as tf
import numpy as np
from vizdoom import *     # Doom Environment
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt


class Preprocessing():

    def __init__(self):
        self.stacked_frames = deque(
            [np.zeros((84, 84), dtype=np.int)
             for i in range(stack_size)], maxlen=4)
        self.stacked_state = np.stack(self.stacked_frames, axis=2)

    def Crop_Screen(self, frame):
        # remove the roof and normalize pixel values
        cropped_frame = frame[30:-10, 30:-30] / 255.0

        cropped_frame = transform.resize(cropped_frame, [84, 84])

        return cropped_frame

    def stack_frames(self, state, is_new_episode):

        cropped_frame = self.Crop_Screen(state)

        if is_new_episode:
            # clear stacked frames
            self.stacked_frames = deque(
                [np.zeros((84, 84), dtype=np.int)
                 for i in range(stack_size)], maxlen=4)
            self.stacked_frames.append(cropped_frame)
            self.stacked_frames.append(cropped_frame)
            self.stacked_frames.append(cropped_frame)
            self.stacked_frames.append(cropped_frame)

            self.stacked_state = np.stack(self.stacked_frames, axis=2)
        else:
            self.stacked_frames.append(cropped_frame)
            self.stacked_state = np.stack(self.stacked_frames, axis=2)

        return self.stacked_state, self.stacked_frames


class DeepQNetwork():

    def __init__(self, state_size, action_size,
                 learning_rate, total_episodes, max_steps,
                 explore_start, explore_stop, name='DeepQNetwork'):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.saver = tf.train.Saver()
        self.game, self.possible_actions = self.creatEnv()

        with tf.variable_scope(name):
            # inputs_: [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32,
                                          [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(
                tf.float32, [None, 3], name="actions")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input: 84x84x4
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs_,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training=True,
                epsilon=1e-5,
                name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # --> [20, 20, 32]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training=True,
                epsilon=1e-5,
                name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # --> [9, 9, 64]

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training=True,
                epsilon=1e-5,
                name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            # --> [1152]

            # FC-layer
            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)

            self.pred_Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions_), axis=1)

            # Loss = Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.pred_Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)

    def creatEnv(self):
        game = DoomGame()
        # Load the correct configuration
        game.load_config("basic.cfg")

        # Load the correct scenario (in our case basic scenario)
        game.set_doom_scenario_path("basic.wad")

        game.init()

        # Here our possible actions
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]

        return game, possible_actions

    def predict_action(self):
        tradeoff = np.random.rand()

        explore_probability = explore_stop + \
            (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if (explore_probability > tradeoff):
            action = random.choice(possible_actions)

        else:
            # Get action from Q-network (exploitation)
            Qs = sess.run(
                DQNetwork.output,
                feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

        return action, explore_probability

    def train(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            decay_step = 0
            self.game.init()

            for episode in range(self.total_episodes):
                step = 0
                episode_rewards = []
                self.game.new_episode()
                state = self.game.get_state().screen_buffer
                preprocessed_frames = Preprocessing()
                state, stacked_frames = preprocessed_frames.stack_frames(
                    state, True)

                while step < self.max_steps:
                    step += 1
                    decay_step += 1

                    # predict the action to take
                    action, explore_probability = self.predict_action()
