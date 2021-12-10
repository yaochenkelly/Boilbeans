import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os


def preprocess(file, window_size):
    removed = set()
    id2name = dict()
    with open(file) as f:
        validSet = set()
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                cols = line.split(',')
                for id, name in enumerate(cols):
                    id2name[id] = name
                validSet = set(range(len(cols)))
                validSet.remove(0)
                validSet.remove(1)
                continue
            for j, num in enumerate(line.split(',')):
                if num == '' and j in validSet:
                    validSet.remove(j)
                    removed.add(j)
        print("valid cols:", len(validSet), sorted(list(validSet)))

    lines = []
    with open(file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                continue
            validLine = []
            for j, num in enumerate(line.split(',')):
                if j in validSet:
                    validLine.append(float(num))
            lines.append(validLine)

    open_prices = np.array(lines)[:, 3]

    # normalize
    sequence_data = preprocessing.normalize(lines, norm="l1", axis=0)

    # split in to windows
    inputs = []
    labels = []
    for start in range(len(sequence_data)): #700*6
        end = start + window_size
        if end >= len(sequence_data):
            break
        inputs.append(sequence_data[start:end])
        labels.append([open_prices[end]])
    return tf.constant(inputs, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)


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


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs, feature_size)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    indices = [i for i in range(len(train_inputs))]
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    examples_num = len(train_inputs)
    for start in range(0, examples_num, model.batch_size):
        end = min(start + model.batch_size, examples_num)
        inputs = train_inputs[start:end]
        labels = train_labels[start:end]

        with tf.GradientTape() as tape:
            y, _ = model.call(inputs, None, True)
            loss = model.loss(y, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def predit(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    sum_square_error = 0
    ys = tf.constant([])
    for start in range(0, len(test_inputs), model.batch_size):
        end = min(start + model.batch_size, len(test_inputs))
        inputs = test_inputs[start:end]
        labels = test_labels[start:end]

        y, _ = model.call(inputs, None, False)
        cur_y = tf.reshape(y, [-1])
        ys = tf.concat([ys, cur_y], axis=-1)

        sum_square_error += model.loss(y, labels) * (end - start)
    return ys, sum_square_error/len(test_labels)


def main():
    dir = './results/'
    model_dir = dir + '/models/'
    figure_dir = dir + '/figures/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        os.mkdir(model_dir)
        os.mkdir(figure_dir)

    window_size = 2
    epoches = 10000
    test_num = 100
    inputs, labels = preprocess('../RL_A2C/AME_processed.csv', window_size)
    print(inputs.shape, labels.shape)

    feature_size = len(inputs[0])
    model = RnnModel(feature_size, window_size)
    # model.load_weights()

    train_inputs, train_labels = inputs[:-test_num], labels[:-test_num]
    test_inputs, test_labels = inputs[-test_num:], labels[-test_num:]

    train_errors = []
    test_errors = []
    all_errors = []
    epochss = []
    for epoch in range(epoches):
        train(model, train_inputs, train_labels)
        if epoch % 100 == 0:
            ys, error = predit(model, inputs, labels)
            _, train_error = predit(model, train_inputs, train_labels)
            test_ys, test_error = predit(model, test_inputs, test_labels)
            print('epoch {}, error {}, train {}, test {}'.format(epoch, round(float(error), 2), round(float(train_error), 2), round(float(test_error), 2)))
            train_errors.append(train_error)
            test_errors.append(test_error)
            all_errors.append(error)
            epochss.append(epoch)

            plt.plot(range(len(labels)), tf.reshape(labels, [-1]))
            plt.plot(range(len(ys)), ys)
            plt.plot([len(labels)-test_num, len(labels)-test_num], [min(tf.reshape(labels, [-1])), max(tf.reshape(labels, [-1]))])
            plt.legend(['y_true', 'y'])
            plt.title('epoch {}, error {}, train {}, test {}'.format(epoch, round(float(error), 2), round(float(train_error), 2), round(float(test_error), 2)))
            plt.xlabel('time')
            plt.ylabel('price')
            plt.savefig(figure_dir + '/epoch' + str(epoch) + '.png')
            plt.show()

            save_path = model_dir + '/epoch' + str(epoch) + '_error' + str(round(float(error), 2)) + '_train' + str(round(float(train_error), 2)) + '_test' + str(round(float(test_error), 2))
            model.save_weights(save_path)

    plt.plot(epochss[:30], train_errors[:30])
    plt.plot(epochss[:30], test_errors[:30])
    plt.plot(epochss[:30], all_errors[:30])
    plt.title("Train, test and all loss v.s. epoch")
    plt.legend(['train', 'test', 'all'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(figure_dir + '/lossFigure.png')
    plt.show()

if __name__ == '__main__':
    main()
