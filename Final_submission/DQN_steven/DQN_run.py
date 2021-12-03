import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN_Model import Agent
# from Preprocess import preprocess
from Preprocess import preprocess_data_v2

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "num_epochs": 1000,
    "batch_size": 128,
    "lr": 0.0002,
    "start_fund": 1000.0,
    "trading_length": 10,    # days to trade and change funding
    "window_size": 1,       # days to put in the model to analyse
    "trade_amount": 3,       # stocks that bot can by at a time
    "trade_money": 200,       # stocks that bot can by at a time
    "num_actions": 6         # 0 = buy 10%, 1 = buy 30%, 2 = sell 10%, 3 = sell 30%, 4 = hold, 5 = clear stock
}

def transition(state, action, price, price_):   # price is the current price; price_ is the yesterday's price

    new_state = state
    # udpate fund first
    new_state[0] += state[1] * (price - price_)

    # update stock hold after
    if action == 0:     # buy
        if (state[0]-state[1]*price) > (hyperparams["start_fund"]*0.1):     # if cash is sufficient
            new_state[1] += hyperparams["start_fund"]*0.1/price
    elif action == 1:
        if (state[0]-state[1]*price) > (hyperparams["start_fund"]*0.3):     # if cash is sufficient
            new_state[1] += hyperparams["start_fund"]*0.3/price
    elif action == 2:     # sell 10%
        new_state[1] = max((state[1] - hyperparams["start_fund"]*0.1/price),0)
    elif action == 3:     # sell 30%
        new_state[1] = max((state[1] - hyperparams["start_fund"]*0.3/price),0)
    elif action == 5:     # clear stock hold
        new_state[1] = 0
    
    return new_state

def get_start_state():
    return np.array([hyperparams["start_fund"],0])      # state = [fund total, number of stocks owned]


def is_terminal_state(state):
    if state[0] < 10:
        return True
    else:
        return False

def get_training_data(stock_array):

    start_day = np.random.randint(0, stock_array.shape[0] - hyperparams["trading_length"] - hyperparams["window_size"] - 2)

    observation = stock_array[start_day:(start_day+hyperparams["window_size"]),:].reshape(-1)
    trading_data = stock_array[(start_day+hyperparams["window_size"]):(start_day+hyperparams["window_size"]+hyperparams["trading_length"]),:].reshape(-1)

    return trading_data, observation

if __name__ == '__main__':

    # preprocess data
    info_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_mfi', 'volume_em', 
                'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap', 'volatility_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
                'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_kcw', 'volatility_kcp', 'volatility_dcl', 'volatility_dch', 'volatility_dcm', 'volatility_dcw', 
                'volatility_dcp', 'volatility_ui', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx', 
                'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 
                'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a', 
                'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind']

    stock = 'AAPL'
    # observation_init, trading_data_init = preprocess(stock, info_list, hyperparams["window_size"], hyperparams["trading_length"])
    # print('observation_init.shape',observation_init.shape)
    # print('trading_data_init.shape',trading_data_init.shape)
    stock_array = preprocess_data_v2(stock, info_list)
    print('stock_array.shape',stock_array.shape)

    # initialize agent
    input_dim = hyperparams["window_size"] * len(info_list) + 2
    agent = Agent(gamma=0.99, epsilon=1.0, lr=hyperparams["lr"], input_dim=input_dim, batch_size=hyperparams["batch_size"], num_actions=hyperparams["num_actions"], eps_end=0.01, device=device)   # eps_end=0.01
    scores, eps_history = [], []

    start_time = time.time()

    # train
    for i in range(hyperparams["num_epochs"]):
        done = False
        state = get_start_state()
        # trading_data = trading_data_init
        # observation = observation_init
        trading_data, observation = get_training_data(stock_array)
        today_data = trading_data[0:len(info_list)]
        price = today_data[3]
        score = 0
        day_cnt = 0
        while not done:
            # get the history data
            observation_with_state = np.append(state,observation)
            action = agent.choose_action(observation_with_state)

            # transition
            today_data = trading_data[0:len(info_list)]
            price_ = price          # yesterday's price
            price = today_data[3]   # today's price
            fund_tmp = state[0]
            state = transition(state, action, price, price_)
            done = is_terminal_state(state)
            if done == True:
                break

            reward = state[0] - fund_tmp
            score += reward

            # udpate observation
            observation_ = np.append(observation[len(info_list):],today_data)
            observation_with_state_ = np.append(state,observation_)
            trading_data = trading_data[len(info_list):]

            # learn
            agent.store_transition(observation_with_state, action, reward, observation_with_state_, done)
            agent.learn()
            
            # update next round
            observation = observation_
            day_cnt += 1
            if day_cnt == hyperparams["trading_length"]:
                done = True

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-10:])

        if i % 10 == 0:
            print('episode',i,'score %.6f' % score,'avg_score %.6f' % avg_score, 'epsilon %.2f' %agent.epsilon)


    # moving_avg = scores
    # for i in range(len(scores)-10):
    #     moving_avg[int(i)+10] = np.mean(scores[int(i):int(i)+10])
    

    # print('scores',scores)
    # print('moving_avg',moving_avg)

    with open('./scores_4.npy','wb') as f:
        np.save(f, scores)

    end_time = time.time()

    print('time taken:',end_time-start_time)

    x = [i+1 for i in range(hyperparams["num_epochs"])]
    # plt.plot(x, moving_avg, label = '10-day moving average')
    plt.plot(x, scores, label = 'scores')
    plt.xlabel('number of games')
    plt.ylabel('score')
    plt.grid()
    plt.legend()
    plt.savefig('score_plot.png')
    plt.show()
