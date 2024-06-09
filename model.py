"""
The file of actor and critic architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from parameters import EPISODE_LEN


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.core = nn.LSTM(256, 256, 2, batch_first=True)
        self.fc2 = nn.Linear(256, a_dim)
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.activ = nn.ELU()

    def forward(self, s, hs_cs):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            logits for each action
        """
        s1, s2 = s

        s1 = self.activ(self.fc1(s1))
        s1, hs_cs = self.core(s1, hs_cs)
        s1 = self.fc2(s1)
        return s1, hs_cs

    def choose_action(self, s, hs_cs=None, device="cpu"):
        """
        Choose action by normal distribution

        Args:
            s: current observation
        Returns:
            probability for each action
        """
        if hs_cs is None:
            hs = torch.zeros((2, len(s), 256), dtype=torch.float32).to(device)
            cs = torch.zeros((2, len(s), 256), dtype=torch.float32).to(device)
            hs_cs = (hs, cs)
        with torch.no_grad():
            self.eval()
            action_prob_ts, hs_cs = self.forward(s, hs_cs)
            action_prob_ts = torch.softmax(action_prob_ts, dim=2)
            action_prob_np = action_prob_ts.squeeze().cpu().numpy()
        return action_prob_np, hs_cs


class Critic(nn.Module):
    """
    Critic network
    """

    def __init__(self, s_dim):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.fc1 = nn.Linear(s_dim, 256)
        self.core = nn.LSTM(256, 256, 2, batch_first=True)
        self.fc2 = nn.Linear(256, 1)
        self.activ = nn.ELU()

    def forward(self, s, hs_cs):
        """
        Network forward function.

        Args:
            s: curent observation
        Returns:
            estimated value of current state
        """
        s1, s2 = s

        s1 = self.activ(self.fc1(s1))
        s1, hs_cs = self.core(s1, hs_cs)
        s1 = self.fc2(s1)
        return s1, hs_cs
