import numpy as np
import csv
import pandas as pd
import os
from sklearn import preprocessing

def preprocess(stock, keep_col, analysis_size, trading_size):

    # get the desired data
    # direction = '../CS2951F_Project/data/processed/price_history_2018_6_2_to_2021_6_4_interval_1d/'
    direction = './'
    filename = direction + stock + '_processed.csv'
    f = pd.read_csv(filename)
    new_f = f[keep_col]
    tmp_file = stock+"_tmp.csv"
    new_f.to_csv(tmp_file, index=False)

    # stock_array = np.loadtxt(open(tmp_file, "rb"), delimiter=",", skiprows=40)
    stock_array = np.genfromtxt (tmp_file, delimiter=",")
    os.remove(tmp_file)
    # print('stock_array',stock_array)

    observation = stock_array[-(analysis_size+trading_size):-trading_size,:].reshape(-1)
    trading_data = stock_array[-trading_size:,:].reshape(-1)

    observation_normalize = preprocessing.normalize(observation, norm="l1", axis=0)
    trading_data_normalize = preprocessing.normalize(trading_data, norm="l1", axis=0)

    return observation_normalize, trading_data_normalize


def preprocess_data_v2(stock, keep_col):
    """
    This function preproccecss the stock history

    :returns:
    open_price: np array of the open price of each day
    data_normalize: np array of normalized features for each day

    """
    # get the desired data
    # direction = '../CS2951F_Project/data/processed/price_history_2018_6_2_to_2021_6_4_interval_1d/'
    direction = './'
    filename = direction + stock + '_processed.csv'
    f = pd.read_csv(filename)
    new_f = f[keep_col]
    tmp_file = stock+"_tmp.csv"
    new_f.to_csv(tmp_file, index=False)

    # stock_array = np.loadtxt(open(tmp_file, "rb"), delimiter=",", skiprows=40)
    stock_array = np.genfromtxt (tmp_file, delimiter=",")
    os.remove(tmp_file)


    stock_array = stock_array[60:,:]
    # normalize the data
    stock_array_normalize = preprocessing.normalize(stock_array, norm="l1", axis=0)

    return stock_array_normalize