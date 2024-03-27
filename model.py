"""
The file of actor and critic architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import choice


class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5,
                               stride=5, padding=0)
        self.bn2d_1 = nn.BatchNorm2d(num_features=10)
        self.bn2d_2 = nn.BatchNorm2d(num_features=10)
        self.bn2d_3 = nn.BatchNorm2d(num_features=10)

        self.fc1 = nn.Linear(s_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256 + 250, 128)
        self.fc4_1 = nn.Linear(128, 5)
        self.fc4_2 = nn.Linear(128, 9)
        self.bn1d_1 = nn.BatchNorm1d(512)
        self.bn1d_2 = nn.BatchNorm1d(256)
        self.bn1d_3 = nn.BatchNorm1d(128)
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

        s2 = self.bn2d_1(self.conv1(s2))
        s2 = F.max_pool2d((F.relu(s2)), kernel_size=2)
        s2 = self.bn2d_2(self.conv2(s2))
        s2 = F.max_pool2d((F.relu(s2)), kernel_size=2)
        s2 = self.bn2d_3(self.conv3(s2))
        s2 = F.max_pool2d((F.relu(s2)), kernel_size=2)
        s2 = s2.view(-1, 5 * 5 * 10)

        s1 = self.bn1d_1(torch.tanh(self.fc1(s1)))
        s1 = self.bn1d_2(torch.tanh(self.fc2(s1)))

        s = torch.concatenate((s1, s2), dim=1)
        s = self.bn1d_3(torch.tanh(self.fc3(s)))

        return self.fc4_1(s), self.fc4_2(s)

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
            action_button_prob_ts, action_stick_prob_ts = self.forward(s)
            action_button_prob_ts = torch.softmax(action_button_prob_ts, dim=1)
            action_stick_prob_ts = torch.softmax(action_stick_prob_ts, dim=1)
            action_button_prob_np = action_button_prob_ts.squeeze().cpu().numpy()
            action_stick_prob_np = action_stick_prob_ts.squeeze().cpu().numpy()

            # a = torch.argmax(action_prob).item()
            a_button = choice(list(range(5)),
                              1,
                              replace=False,
                              p=action_button_prob_np)[0]
            a_stick = choice(list(range(9)),
                             1,
                             replace=False,
                             p=action_stick_prob_np)[0]
        return (a_button, a_stick), (action_button_prob_np, action_stick_prob_np)


class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, s_dim):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5,
                               stride=5, padding=0)
        self.bn2d_1 = nn.BatchNorm2d(num_features=10)
        self.bn2d_2 = nn.BatchNorm2d(num_features=10)
        self.bn2d_3 = nn.BatchNorm2d(num_features=10)

        self.fc1 = nn.Linear(s_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256 + 250, 1)
        self.bn1d_1 = nn.BatchNorm1d(512)
        self.bn1d_2 = nn.BatchNorm1d(256)
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

        s2 = self.bn2d_1(self.conv1(s2))
        s2 = F.max_pool2d((F.relu(s2)), kernel_size=2)
        s2 = self.bn2d_2(self.conv2(s2))
        s2 = F.max_pool2d((F.relu(s2)), kernel_size=2)
        s2 = self.bn2d_3(self.conv3(s2))
        s2 = F.max_pool2d((F.relu(s2)), kernel_size=2)
        s2 = s2.view(-1, 5 * 5 * 10)

        s1 = self.bn1d_1(torch.tanh(self.fc1(s1)))
        s1 = self.bn1d_2(torch.tanh(self.fc2(s1)))

        s = torch.concatenate((s1, s2), dim=1)

        return self.fc3(s)
