"""
The file of actor and critic architectures.
"""
import torch
import torch.nn as nn
from numpy.random import choice


class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, a_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.a_dim = a_dim
        self.set_init([self.fc1, self.fc2, self.fc2])

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
        x = self.bn1(torch.tanh(self.fc1(s)))
        x = self.bn2(torch.tanh(self.fc2(x)))
        return self.fc3(x)

    def choose_action(self, s):
        """
        Choose action by normal distribution

        Args:
            s: current observation
        Returns:
            action tensor sampled from policy(normal distribution),
            log probability of the action
        """
        action_prob = torch.softmax(self.forward(s), dim=1).squeeze()
        # a = torch.argmax(action_prob).item()
        a = choice(list(range(self.a_dim)), 1, replace=False, p=action_prob.cpu().numpy())[0]
        return a, action_prob


class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, s_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
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
        x = self.bn1(torch.tanh(self.fc1(s)))
        x = self.bn2(torch.tanh(self.fc2(x)))
        values = self.fc3(x)
        return values
