import numpy as np
from sklearn import preprocessing
from model import ReinforceWithBaseline, RnnModel
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def discount(rewards, discount_factor=.99):
    discounted_rewards = [rewards[-1]]
    for i in reversed(range(len(rewards)-1)):
        discounted_rewards.append(rewards[i] + discount_factor * discounted_rewards[-1])
    return discounted_rewards[::-1]


def preprocess(file, window_size):
    with open(file) as f:
        validSet = set()
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                cols = line.split(',')
                validSet = set(range(len(cols)))
                validSet.remove(0)
                validSet.remove(1)
                continue
            for j, num in enumerate(line.split(',')):
                if num == '' and j in validSet:
                    validSet.remove(j)
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

    open_prices = np.array(lines)[:, 0]

    # normalize
    sequence_data = preprocessing.normalize(lines, norm="l1", axis=0)

    # split in to windows
    inputs = []
    labels = []
    for start in range(len(sequence_data)):
        end = start + window_size
        if end >= len(sequence_data):
            break
        inputs.append(sequence_data[start:end])
        labels.append([open_prices[end]])
    return tf.constant(inputs, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)


def transition(money_init, money_all, money_stock, action, stock_price_yesterday, stock_price_today):
    trade_ratio_1 = 0.1
    trade_ratio_2 = 0.3

    money_remain = money_all - money_stock
    money_stock = money_stock / stock_price_yesterday * stock_price_today
    money_stock_change = 0
    # sell stock 10%
    if action == 1:
        money_stock_change = -min(money_init * trade_ratio_1, money_stock)
    # sell stock 30%
    if action == 2:
        money_stock_change = -min(money_init * trade_ratio_2, money_stock)
    # buy stock 10%
    if action == 3:
        money_stock_change = min(money_init * trade_ratio_1, money_remain)
    # buy stock 30%
    if action == 4:
        money_stock_change = min(money_init * trade_ratio_2, money_remain)
    money_remain -= money_stock_change
    money_stock += money_stock_change

    return money_stock + money_remain, money_stock


def norm(min, max, cur):
    return (cur - min) / (max - min)


def generate_trajectory(money, inputs, labels, deal_range, rl_model, min_labels, max_labels, rnn_model=None, training=None, start_day=None):
    """
    Generates lists of states, actions, and rewards for a specified day duration

    :returns:
    A tuple of lists (states, actions, rewards), where each list has length equal to the number of days
    """

    # initialize
    money_stock = 0
    money_all = money
    states = []
    actions = []
    rewards = []

    # pick a random start day
    if not start_day:
        start_day = np.random.randint(0, len(labels) - deal_range - 1)
    norm_labels = norm(min_labels, max_labels, labels)

    # start the episode
    for i in range(deal_range):

        # index the current day
        day_ind = start_day + i
        today_feature = inputs[day_ind, :][0]

        # get the current stock price
        stock_price_today = norm_labels[day_ind]
        if rnn_model:
            stock_price_tomorrow, _ = rnn_model.call(tf.reshape(today_feature, [1, 1, len(today_feature)]), None)
            stock_price_tomorrow = tf.squeeze(stock_price_tomorrow)
            stock_price_tomorrow = norm(min_labels, max_labels, stock_price_tomorrow)
            state = tf.concat((np.squeeze([money_all/money, money_stock/money, stock_price_today, stock_price_tomorrow]), today_feature), axis=-1)
            # state = [money_all/money, money_stock/money, stock_price_today, stock_price_tomorrow]
        else:
            state = tf.concat((np.squeeze([money_all/money, money_stock/money, stock_price_today]), today_feature), axis=-1)

        action_prob = tf.squeeze(rl_model.call(tf.expand_dims(state, axis=0), train=training))
        action_prob = np.asarray(action_prob).astype('float64')
        action_prob = action_prob / np.sum(action_prob)
        if training:
            action = np.random.choice(5, 1, p=action_prob)[0]
        else:
            action = np.argmax(action_prob)

        money_all, money_stock = transition(money, money_all, money_stock, action, labels[day_ind-1], labels[day_ind])

        # update
        if i == deal_range - 1:
            reward = money_all - money
        else:
            reward = 0

        actions.append(action)
        rewards.append(reward)
        states.append(state)

    # convert all lists to tensors
    actions = np.array(actions)
    rewards = np.array(rewards)
    states = np.array(states)

    return states, actions, rewards


def train(money, inputs, labels, min_labels, max_labels, deal_range, rl_model, rnn_model):
    final_rewards = []
    losses = []
    episodes = 10000
    for episode in range(episodes):
        with tf.GradientTape() as tape:
            states, actions, rewards = generate_trajectory(money, inputs, labels, deal_range, rl_model, min_labels, max_labels, rnn_model, training=True)
            # discounted_rewards = discount(rewards)
            rewards = discount(rewards)
            loss = rl_model.loss(np.array(states), np.array(actions), np.array(rewards), train=True)
        losses.append(loss)
        gradients = tape.gradient(loss, rl_model.trainable_variables)
        rl_model.optimizer.apply_gradients(zip(gradients, rl_model.trainable_variables))
        final_rewards.append(rewards[-1])

        if episode % (episodes/10) == 0:
            print(episode, rewards[-1])

    # plot the reward history
    plt.figure()
    plt.plot(running_mean(final_rewards, 20))
    plt.plot(np.zeros((len(running_mean(final_rewards, 20)))), 'r-')
    plt.xlabel('episode')
    plt.ylabel('profits')
    plt.title('Actor-Critic Stock Trade Bot 10-day')
    plt.show()

    return losses


def test_rl(money, inputs, labels, deal_range, rl_model, min_labels, max_labels, rnn_model):
    sum_rewards = 0
    count = 0
    final_rewards = []
    for i in range(0, len(labels)-deal_range-1):
        states, actions, rewards = generate_trajectory(money, inputs, labels, deal_range, rl_model, min_labels, max_labels, rnn_model, training=False, start_day=i)
        sum_rewards += rewards[-1]
        count += 1
        final_rewards.append(rewards[-1])
        print(actions)

if __name__ == '__main__':
    # set the initial money and trading days
    money = 1000
    deal_range = 10
    window_size = 1
    num_actions = 5

    # import the datasheet
    file = 'AME_processed.csv'
    inputs, labels = preprocess(file, window_size)
    test_num = 100
    train_inputs, train_labels = inputs[:-test_num], labels[:-test_num]
    test_inputs, test_labels = inputs[-test_num:], labels[-test_num:]

    # initialize the model
    state_size = inputs.shape[1]
    rl_model = ReinforceWithBaseline(state_size, num_actions)
    rnn_model = RnnModel(state_size, window_size)
    # rnn_model.load_weights('../LSTM/results/models/epoch8600_error2.51_train2.45_test2.91')
    rnn_model = None

    # train without
    # train with rnn
    min_labels = min(train_labels)
    max_labels = max(train_labels)
    train(money, train_inputs, train_labels, min_labels, max_labels, deal_range, rl_model, rnn_model=rnn_model)

    # test
    rl_model.load_weights(dir + '/without_rnn')
    mean_earns = test_rl(money, test_inputs, test_labels, deal_range, rl_model, min_labels, max_labels, rnn_model=rnn_model)
    print(mean_earns)


