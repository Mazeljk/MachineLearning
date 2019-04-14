"""
：Realize AI-player in Doom's basic scene(only three actions:left, right, shot)
  with deep Q-learning
：dependencies: python 3.7.1, vizdoom 1.1.7
"""
import tensorflow as tf
import numpy as np
from vizdoom import *     # Doom Environment
import random
from skimage import transform
from collections import deque


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


class ReplayMemory():

    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class DeepQNetwork():

    def __init__(self, state_size, action_size,
                 learning_rate, total_episodes, max_steps,
                 explore_start, explore_stop, decay_rate, name='DeepQNetwork'):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.memory = ReplayMemory()
        self.saver = tf.train.Saver()
        self.game, self.possible_actions = self.creatEnv()
        tf.reset_default_graph()
        self.Network = self._init_DeepQ(state_size, action_size, learning_rate)

    def _init_DeepQ(self, state_size, action_size, learning_rate):

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

    def predict_action(self, decay_step, state):
        tradeoff = np.random.rand()

        explore_probability = self.explore_stop + \
            (self.explore_start - self.explore_stop) * \
            np.exp(-self.decay_rate * decay_step)

        if (explore_probability > tradeoff):
            action = random.choice(self.possible_actions)

        else:
            # Get action from Q-network (exploitation)
            Qs = sess.run(
                self.Network.output,
                feed_dict={self.Network.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value
            choice = np.argmax(Qs)
            action = self.possible_actions[int(choice)]

        return action

    def train(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            decay_step = 0
            self.game.init()
            preprocessed_frames = Preprocessing()

            for episode in range(self.total_episodes):
                step = 0
                self.game.new_episode()
                state = self.game.get_state().screen_buffer
                state, stacked_frames = preprocessed_frames.stack_frames(
                    state, True)

                while step < self.max_steps:
                    step += 1
                    decay_step += 1

                    # predict the action to take
                    action = self.predict_action(decay_step, state)
                    reward = game.make_action(action)
                    done = game.is_episode_finished()

                    if done:
                        next_state = np.zeros((84, 84), dtype=np.int)
                        next_state, stacked_frames = preprocessed_frames.stack_frames(
                            next_state, False)
                        step = self.max_steps
                        memory.add((state, action, reward, next_state, done))
                    else:
                        next_state = game.get_state().screen_buffer
                        next_state, stacked_frames = preprocessed_frames.stack_frames(
                            next_state, False)
                        memory.add((state, action, reward, next_state, done))
                        state = next_state

                # LEARNING PART
                # Obtain random mini-batch from memory
                batch = self.memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(self.Network.output, feed_dict={
                                         self.Network.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set
                # Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    if dones_mb[i]:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * \
                            np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([self.Network.loss, self.Network.optimizer],
                                   feed_dict={
                                   self.Network.inputs_: states_mb,
                                   self.Network.target_Q: targets_mb,
                                   self.Network.actions_: actions_mb})

            # Save model every 5 episodes
            if episode % 5 == 0:
                saver.save(sess, "./models/model.ckpt")
            self.game.close()

    def Play(self):
        with tf.Session() as sess:
            self.saver.restore(sess, "./models/model.ckpt")
            self.game.init()
            done = False
            self.game.new_episode()
            state = self.game.get_state().screen_buffer
            preprocessed_frames = Preprocessing()
            state, stacked_frames = preprocessed_frames.stack_frames(
                state, True)
            while not self.game.is_episode_finished():
                Qs = sess.run(self.Network.output,
                              feed_dict={
                                  self.Network.inputs_:
                                  state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]

                self.game.make_action(action)
                done = self.game.is_episode_finished()
                score = self.game.get_total_reward()

                if done:
                    break
                else:
                    next_state = self.game.get_state().screen_buffer
                    next_state, stacked_frames = preprocessed_frames.stack_frames(
                        next_state, False)
                    state = next_state

            score = self.game.get_total_reward()
            print("Score: ", score)
            self.game.close()
