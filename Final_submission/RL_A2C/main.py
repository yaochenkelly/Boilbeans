import numpy as np
from sklearn import preprocessing
from model import A2C
import torch
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def preprocess_data(data_file):
    """
    This function preproccecss the stock history

    :returns:
    open_price: np array of the open price of each day
    data_normalize: np array of normalized features for each day

    """
    with open(data_file) as f:
        lines = (line for line in f if not line.startswith('#'))
        FH = np.loadtxt(lines, delimiter=',', skiprows=80, usecols=range(2,82))
        # FH = np.loadtxt(lines, delimiter=',', skiprows=80, usecols=range(2,6))

    open_price = FH[:,0]

    # normalize the data
    feature_normalize = preprocessing.normalize(FH, norm="l1", axis=0)

    return open_price, feature_normalize


def generate_trajectory(initial_money, open_price, feature_normalize, days, model):
    """
    Generates lists of states, actions, and rewards for a specified day duration

    :returns: 
    A tuple of lists (states, actions, rewards), where each list has length equal to the number of days
    """    

    # initialize
    stock_curr = 0
    money_curr = initial_money
    states = []
    actions = []
    rewards = []

    # pick a random start day
    start_day = np.random.randint(0, len(open_price)-days-1)

    # start_day = 0
    # print('start_day: '+str(start_day))

    # define 3 actions
    # 0 -> sell stock that worths 10% of the initial money
    # 1 -> do nothing
    # 2 -> buy stock that worths 10% of the initial money

    # start the episode
    for i in range(days):

        # index the current day
        day_ind = start_day + i

        # geet the current stock price
        stock_price_curr = open_price[day_ind]

        # # use random action for now
        # action = np.random.randint(3)

        # always buy at the first day
        if i == 0:
            action = 3

        else:
            # action from the A2C model
            action_prob = model.policy_single_states(torch.FloatTensor(state)).cpu().detach().numpy()
            action_prob /= np.sum(action_prob)
            action = np.random.choice(5,1,p=action_prob)[0]

        # print('action: '+str(action))

        # set the trade ratio
        trade_ratio_1 = 0.1
        trade_ratio_2 = 0.3

        # sell stock 10%
        if action == 1:
            stock_num = trade_ratio_1 * initial_money / stock_price_curr
            stock_to_sell = min(stock_curr, stock_num)
            stock_curr -= stock_to_sell
            money_curr += stock_to_sell * stock_price_curr

        # sell stock 30%
        if action == 2:
            stock_num = trade_ratio_2 * initial_money / stock_price_curr
            stock_to_sell = min(stock_curr, stock_num)
            stock_curr -= stock_to_sell
            money_curr += stock_to_sell * stock_price_curr

        # buy stock
        if action == 3:
            money_to_buy = min(trade_ratio_1 * initial_money, money_curr)
            stock_num = money_to_buy / stock_price_curr
            stock_curr += stock_num
            money_curr -= money_to_buy

        # buy stock
        if action == 4:
            money_to_buy = min(trade_ratio_2 * initial_money, money_curr)
            stock_num = money_to_buy / stock_price_curr
            stock_curr += stock_num
            money_curr -= money_to_buy

        # print('stoch_price_curr: '+str(stock_price_curr))
        # print('stock_curr: '+str(stock_curr))
        # print('money_curr: '+str(money_curr))

        # update
        actions.append(action)
        reward = money_curr + stock_curr * stock_price_curr - initial_money
        rewards.append(reward)
        # print('reward: '+str(reward))

        state = np.hstack((np.array([money_curr, stock_curr]), feature_normalize[day_ind, :]))
        states.append(state)

        # print('-----------------------')

    # convert all lists to tensors
    actions = torch.IntTensor(actions)
    rewards = torch.FloatTensor(rewards)
    states  = torch.FloatTensor(states)

    # print('actions:')
    # print(actions)
    # print('rewards')
    # print(rewards)
    # print('states')
    # print(states)

    return states, actions, rewards


def train(initial_money, open_price, feature_normalize, days, model):
    """
    Model training
    """  

    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    final_reward = []

    for episode in range(1000):

        # generate the trajectories and calculate the loss
        states, actions, rewards = generate_trajectory(initial_money, open_price, feature_normalize, days, model)
        loss = model.loss(states, actions, rewards)

        # compute the gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('loss'+str(loss))
        final_reward.append(rewards[-1])

    # plot the reward history
    plt.figure()
    plt.plot(running_mean(final_reward, 20))
    plt.plot(np.zeros((len(running_mean(final_reward, 20)))), 'r-')
    plt.xlabel('episode')
    plt.ylabel('profits')
    plt.title('Actor-Critic Stock Trade Bot 10-day')
    plt.show()


if __name__ == '__main__':
    
    # import the datasheet
    stock_history_chart = 'TSLA_processed.csv'
    open_price, feature_normalize = preprocess_data(stock_history_chart)
    
    # print(feature_normalize)
    # print(open_price/np.sum(open_price))

    # set the initial money and trading days
    initial_money = 1000
    days = 10

    # initialize the model
    state_size = 82
    num_actions = 5
    model = A2C(state_size, num_actions)

    # generate
    train(initial_money, open_price, feature_normalize, days, model)


