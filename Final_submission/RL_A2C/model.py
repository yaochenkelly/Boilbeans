import torch
import torch.nn.functional as F


class A2C(torch.nn.Module):
    def __init__(self, state_size, num_actions):
    	super().__init__()

    	# define the parameters and layers
    	self.num_actions = num_actions
    	self.actor_layer1_size = 50
    	self.critic_layer1_size = 50

    	# actor layer
    	self.actor_dense1 = torch.nn.Linear(state_size, self.actor_layer1_size)
    	self.actor_dense2 = torch.nn.Linear(self.actor_layer1_size, num_actions)

    	# critic layer
    	self.critic_dense1 = torch.nn.Linear(state_size, self.critic_layer1_size)
    	self.critic_dense2 = torch.nn.Linear(self.actor_layer1_size, 1)

    	# soft max for policy
    	self.softmax_d0 = torch.nn.Softmax(dim=0)
    	self.softmax_d1 = torch.nn.Softmax(dim=1)


    def policy_single_states(self, states):

    	output = self.actor_dense1(states)
    	output = self.actor_dense2(output)
    	output = self.softmax_d0(output)

    	return output


    def policy_batch_states(self, states):

    	output = self.actor_dense1(states)
    	output = self.actor_dense2(output)
    	output = self.softmax_d1(output)

    	return output


    def value_function(self, states):

    	output = self.critic_dense1(states)
    	output = self.critic_dense2(output)

    	return output


    def loss(self, states, actions, rewards):

    	policy = self.policy_batch_states(states)
    	value = self.value_function(states)

    	# print('---policy---')
    	# print(policy)
    	# print('---value---')
    	# print(value)

    	ind = torch.stack([torch.arange(len(actions)), actions], axis=1)
    	# print('---index---')
    	# print(ind)

    	Prob_a_s = policy[list(ind.T)]
    	# print('---Prob_a_s---')
    	# print(Prob_a_s)

    	# reshape rewards to match the shape of value
    	rewards = rewards.reshape(value.size())
    	# print('---rewards---')
    	# print(rewards)

    	loss_actor = -torch.sum(torch.log(Prob_a_s) * (rewards - value.detach()))
    	loss_critic = torch.sum(torch.square(rewards - value))

    	# print('--- reward - value ---')
    	# print(rewards - value)

    	return loss_actor + loss_critic



