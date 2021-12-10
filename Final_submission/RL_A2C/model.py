import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        self.hidden_size = 128
        self.q1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.q2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.drop2 = tf.keras.layers.Dropout(0.2)
        self.q3 = tf.keras.layers.Dense(self.num_actions, activation='softmax')
        self.v1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(0.2)
        self.v2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.drop4 = tf.keras.layers.Dropout(0.2)
        self.v3 = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def call(self, states, train=None):
        out = self.q1(states)
        if train:
            out = self.drop1(out)
        out = self.q2(out)
        if train:
            out = self.drop2(out)
        out = self.q3(out)
        return out

    def value_function(self, states, train=None):
        out = self.v1(states)
        if train:
            out = self.drop3(out)
        out = self.v2(out)
        if train:
            out = self.drop4(out)
        out = self.v3(out)
        return out

    def loss(self, states, actions, discounted_rewards, train=None):
        state_values = tf.squeeze(self.value_function(states, train=train))
        A = tf.stop_gradient(tf.cast(discounted_rewards - state_values, dtype=tf.float32))

        action_probs = self.call(states, train=train)
        actions = tf.expand_dims(actions, axis=1)
        log_y = tf.math.log(tf.gather_nd(action_probs, actions, batch_dims=1))

        loss1 = - tf.reduce_sum(tf.multiply(log_y, A))
        loss2 = tf.reduce_sum(tf.cast(discounted_rewards - state_values, dtype=tf.float32) ** 2)
        return loss1 + loss2


class RnnModel(tf.keras.Model):
    def __init__(self, feature_size, window_size):
        """
        The Model class predicts the next day's price with window_size history.

        :param feature_size: The number of features in each day's data
        """

        super(RnnModel, self).__init__()
        # initialize vocab_size, embedding_size
        self.feature_size = feature_size
        self.window_size = window_size  # DO NOT CHANGE!
        self.batch_size = 64
        self.d_model = 32
        self.hidden_size = 32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.lstm = tf.keras.layers.LSTM(self.d_model, return_sequences=True, return_state=True)
        self.fc1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.2)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, inputs, initial_state, training=None):
        """
        :param inputs: word ids of shape (batch_size, window_size, feature_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """
        out, last_output, cell_state = self.lstm(inputs, initial_state=initial_state)
        out = self.fc1(last_output)
        if training:
            out = self.drop(out)
        out = self.fc2(out)
        return out, (last_output, cell_state)

    def loss(self, y, y_true):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean((y - y_true) ** 2)


