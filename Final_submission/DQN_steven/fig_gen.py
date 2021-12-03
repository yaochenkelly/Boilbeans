# -*- coding: utf-8 -*-

import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
    
with open('./scores_4.npy','rb') as f:
    scores = np.load(f)

avg_length = 100

moving_avg = np.zeros(len(scores))
for i in range(len(scores)):
    moving_avg[int(i)] = np.mean(scores[max(int(i)-avg_length,0):int(i)])

num_epochs = 1000

x = [i+1 for i in range(num_epochs)]

distance = 1000
# plt.plot(x, scores, label = 'scores')
plt.plot(x[10:distance], moving_avg[10:distance], label = 'moving average')
plt.xlabel('number of games')
plt.ylabel('score')
plt.grid()
plt.legend()
plt.savefig('score_plot.png')
plt.show()