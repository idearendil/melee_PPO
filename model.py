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
        self.fc3_1 = nn.Linear(128, a_dim)
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

        return self.fc3_1(s1)

    def choose_action(self, s):
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
            action_prob_ts = self.forward(s)
            action_prob_ts = torch.softmax(action_prob_ts, dim=1)
            action_prob_np = action_prob_ts.squeeze().cpu().numpy()
        return action_prob_np


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
