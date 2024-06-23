from abc import ABC, abstractmethod
import code
import random
import torch
from melee import enums
from melee.stages import EDGE_POSITION
import numpy as np
from melee_env.agents.util import (
    ObservationSpace,
    ActionSpace,
    MyActionSpace,
    from_observation_space,
    from_action_space,
)
from PPO import Ppo
from melee_env.agents.basic import *


class PPOAgent(Agent):
    """
    An agent using PPO algorithm.
    """

    def __init__(
        self, character, agent_id, opponent_id, device, s_dim, a_dim, test_mode=False
    ):
        super().__init__()
        self.character = character
        self.agent_id = agent_id
        self.opponent_id = opponent_id

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.device = device

        self.action_space = ActionSpace()

        self.prev_action = 0
        self.ppo = Ppo(self.s_dim, self.a_dim, self.device)
        self.test_mode = test_mode

        self.action_q = []
        self.action_q_idx = 0
        self.hs_cs = (
            torch.zeros((2, 1, 256), dtype=torch.float32).to(device),
            torch.zeros((2, 1, 256), dtype=torch.float32).to(device),
        )

    def act(self, s):

        act_data = None

        if self.action_q_idx >= len(self.action_q):
            # the agent should select a new action
            self.action_q_idx = 0
            action_prob_np, self.hs_cs = self.ppo.choose_action(
                (s, self.prev_action), self.agent_id, self.hs_cs
            )

            if self.test_mode:
                # choose the most probable action
                new_action = np.argmax(action_prob_np)
            else:
                # choose an action with probability weights
                new_action = random.choices(
                    list(range(self.a_dim)), weights=action_prob_np, k=1
                )[0]
            if len(self.action_q) > 0:
                self.prev_action = self.action_q[0]
            self.action_q = [new_action, new_action, new_action]
            act_data = new_action

        now_action = self.action_q[self.action_q_idx]
        self.action_q_idx += 1

        return now_action, act_data
