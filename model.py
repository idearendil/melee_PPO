"""
The file of actor and critic architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from parameters import TAU


class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_1 = nn.Linear(128, 5)
        self.fc3_2 = nn.Linear(128, 9)
        self.bn1d_1 = nn.BatchNorm1d(256)
        self.bn1d_2 = nn.BatchNorm1d(128)
        self.a_dim = a_dim
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        """
        Weight initialization method
        """
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            mu and sigma of policy considering current observation
        """
        s1, s2 = s

        s1 = self.bn1d_1(torch.tanh(self.fc1(s1)))
        s1 = self.bn1d_2(torch.tanh(self.fc2(s1)))

        return self.fc3_1(s1), self.fc3_2(s1)

    def choose_action(self, s, test_mode):
        """
        Choose action by normal distribution

        Args:
            s: current observation
        Returns:
            action tensor sampled from policy(normal distribution),
            log probability of the action
        """
        with torch.no_grad():
            self.eval()
            action_button_prob_ts, action_stick_prob_ts = self.forward(s)
            action_button_prob_ts = torch.softmax(action_button_prob_ts, dim=1)
            action_stick_prob_ts = torch.softmax(action_stick_prob_ts, dim=1)
            action_button_prob_np = action_button_prob_ts.squeeze().cpu().numpy()
            action_stick_prob_np = action_stick_prob_ts.squeeze().cpu().numpy()

            if test_mode:
                a_button = torch.argmax(action_button_prob_np).item()
                a_stick = torch.argmax(action_stick_prob_np).item()
            else:
                a_button = self.boltzmann(list(range(5)), action_button_prob_np)
                a_stick = self.boltzmann(list(range(9)), action_stick_prob_np)
        return (a_button, a_stick), (action_button_prob_np, action_stick_prob_np)

    def boltzmann(self, actions, weights):
        """
        Boltzmann greedy exploration method.

        :arg actions:
            tuple of possible actions.

        :arg weights:
            numpy array(float) of weights for each possible action

        :returns:
            chosen action among actions
        """
        # print(weights)
        max_weight = np.max(weights)
        exp_weights = np.exp((weights - max_weight) / TAU)
        sum_exp_weights = np.sum(exp_weights)
        final_weights = exp_weights / sum_exp_weights
        # print(final_weights)
        return random.choices(actions, weights=final_weights, k=1)[0]


class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, s_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.bn1d_1 = nn.BatchNorm1d(256)
        self.bn1d_2 = nn.BatchNorm1d(128)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        """
        Weight initialization method(but not used now)
        """
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, s):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            estimated value of current state
        """
        s1, s2 = s

        s1 = self.bn1d_1(torch.tanh(self.fc1(s1)))
        s1 = self.bn1d_2(torch.tanh(self.fc2(s1)))

        return self.fc3(s1)
