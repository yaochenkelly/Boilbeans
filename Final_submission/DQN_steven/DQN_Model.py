import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device = ', device)

class DQN(nn.Module):

    def __init__(self, lr, input_dim, fc1_dim, fc2_dim, num_actions, device):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim, dtype=torch.float64)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim, dtype=torch.float64)
        self.fc3 = nn.Linear(self.fc2_dim, self.num_actions, dtype=torch.float64)
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()

        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.double()))
        x = F.relu(self.fc2(x.double()))
        actions = self.fc3(x.double())

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dim, batch_size, num_actions, device, max_mem_size=100000, eps_end=0.02, eps_dec=1e-4):       #eps_end=0.01, eps_dec=5e-4
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end      # epsilon threshold
        self.eps_dec = eps_dec      # epsilon step size
        self.lr = lr
        self.action_space = [i for i in range(num_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cnt = 0
        self.device = device

        self.Q_eval = DQN(lr=self.lr,
            input_dim = input_dim, 
            fc1_dim = 1024,
            fc2_dim = 512,
            num_actions = num_actions, 
            device = device)

        # initailize memories
        self.state_mem = np.zeros((self.mem_size, input_dim), dtype = np.float32)
        self.new_state_mem = np.zeros((self.mem_size, input_dim), dtype = np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype = np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cnt % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.reward_mem[index] = reward
        self.action_mem[index] = action
        self.terminal_mem[index] = done

        self.mem_cnt += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation).to(self.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()       # returns a tensor, so needs '.item()'
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # make sure when there are too many 0s in the memory, dont bother learning
        if self.mem_cnt < self.batch_size:
            return      

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_mem[batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.device)
        reward_batch = torch.tensor(self.reward_mem[batch]).to(self.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.device)
        action_batch = self.action_mem[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min