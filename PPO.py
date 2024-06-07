"""
The file of Ppo class.
"""

import torch.optim as optim
import numpy as np
import torch
from torch.distributions import Categorical
from model import Actor, Critic
from parameters import (
    LR_ACTOR,
    LR_CRITIC,
    GAMMA,
    LAMBDA,
    BATCH_SIZE,
    EPSILON,
    L2_RATE,
    BUFFER_SIZE,
    BATCH_NUM,
    ENTROPY_WEIGHT,
    PRE_STATES_NUM,
)
from replay_buffer import ReplayBuffer
from math import log


class Ppo:
    """
    The class which Proximal Policy Optimization is implemented in.
    """

    def __init__(self, s_dim, a_dim, device):
        self.device = device
        self.actor_net = Actor(s_dim * (PRE_STATES_NUM + 1), a_dim).to(self.device)
        self.critic_net = Critic(s_dim * (PRE_STATES_NUM + 1)).to(self.device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), lr=LR_CRITIC, weight_decay=L2_RATE
        )
        self.critic_loss_func = torch.nn.MSELoss()
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.s_dim = s_dim

    def models_to_device(self, device):
        """
        Move actor and critic to the specific device('cpu' or 'cuda:x')
        """
        self.actor_net.to(device)
        self.critic_net.to(device)

    def choose_action(self, s, agent_id):
        """
        Convert state to preprocessed tensor,
        and then return action probability by actor_net
        """
        s_ts1, s_ts2 = self.state_preprocessor(s, agent_id)
        s_ts1 = torch.from_numpy(s_ts1).unsqueeze(0).to(self.device)
        s_ts2 = torch.from_numpy(s_ts2).unsqueeze(0).to(self.device)
        return self.actor_net.choose_action((s_ts1, s_ts2))

    def push_an_episode(self, data, agent_id):
        """
        Push an episode to replay buffer.

        Args:
            data: an array of ([s1, a, a_prob], [s2, r, mask])
        Returns:
            None
        """

        s1_lst, a_lst, prob_lst, s2_lst, r_lst, mask_lst = [], [], [], [], [], []
        for data1, data2 in data:
            s1, a, prob = data1
            s2, r, mask = data2
            s1_lst.append(s1)
            a_lst.append(a)
            prob_lst.append((torch.Tensor(prob)))
            s2_lst.append(s2)
            r_lst.append(r)
            mask_lst.append(mask)

        s1_lst1, s1_lst2, s2_lst1, s2_lst2 = [], [], [], []
        for s1 in s1_lst:
            s1_1, s1_2 = self.state_preprocessor(s1, agent_id)
            s1_lst1.append(s1_1)
            s1_lst2.append(s1_2)
        for s2 in s2_lst:
            s2_1, s2_2 = self.state_preprocessor(s2, agent_id)
            s2_lst1.append(s2_1)
            s2_lst2.append(s2_2)

        s2_ts1 = torch.Tensor(np.array(s2_lst1, dtype=np.float32)).to(self.device)
        s2_ts2 = torch.Tensor(np.array(s2_lst2, dtype=np.float32)).to(self.device)
        r_ts = torch.Tensor(np.array(r_lst, dtype=np.float32))
        masks = torch.Tensor(np.array(mask_lst, dtype=np.float32))

        with torch.no_grad():
            self.critic_net.eval()
            v_lst = []
            for idx in range(0, len(s2_ts1), BATCH_SIZE):
                idx_end = min(idx + BATCH_SIZE, len(s2_ts1))
                v_lst.append(
                    self.critic_net((s2_ts1[idx:idx_end], s2_ts2[idx:idx_end])).cpu()
                )
            v_ts = torch.concatenate(v_lst, dim=0).squeeze()
            ret_ts, adv_ts = self.td_error(r_ts, masks, v_ts)

        if len(s1_lst) != len(s2_lst):
            print("Error occured!!! len(s1_lst) != len(s2_lst)")

        for idx, _ in enumerate(s1_lst1):
            self.buffer.push(
                (
                    s1_lst[idx],
                    s2_lst[idx],
                    a_lst[idx],
                    adv_ts[idx].item(),
                    ret_ts[idx].item(),
                    prob_lst[idx],
                    agent_id,
                )
            )

    def train(self, episode_id):
        """
        Train Actor network and Critic network with data in buffer.
        """
        print("buffer size: ", self.buffer.size())

        self.actor_net.train()
        self.critic_net.train()
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), lr=LR_CRITIC, weight_decay=L2_RATE
        )
        critic_loss_lst, actor_loss_lst, entropy_loss_lst = [], [], []
        for batch_id in range(BATCH_NUM):

            s1_lst, s2_lst, a_lst, adv_lst, ret_lst, op_lst, id_lst = self.buffer.pull(
                BATCH_SIZE
            )

            s1_lst1, s1_lst2, s2_lst1, s2_lst2 = [], [], [], []
            for s1, s2, agent_id in zip(s1_lst, s2_lst, id_lst):
                s1_1, s1_2 = self.state_preprocessor(s1, agent_id)
                s1_lst1.append(s1_1)
                s1_lst2.append(s1_2)
                s2_1, s2_2 = self.state_preprocessor(s2, agent_id)
                s2_lst1.append(s2_1)
                s2_lst2.append(s2_2)

            s1_ts1 = torch.Tensor(np.stack(s1_lst1, axis=0)).to(self.device)
            s1_ts2 = torch.Tensor(np.stack(s1_lst2, axis=0)).to(self.device)
            s2_ts1 = torch.Tensor(np.stack(s2_lst1, axis=0)).to(self.device)
            s2_ts2 = torch.Tensor(np.stack(s2_lst2, axis=0)).to(self.device)
            a_ts = torch.LongTensor(a_lst).to(self.device).unsqueeze(1)
            adv_ts = (
                torch.Tensor(np.array(adv_lst, dtype=np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            ret_ts = (
                torch.Tensor(np.array(ret_lst, dtype=np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            op_ts = torch.Tensor(np.stack(op_lst, axis=0)).to(self.device)

            v_ts = self.critic_net((s2_ts1, s2_ts2))
            critic_loss = self.critic_loss_func(v_ts, ret_ts)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            new_probs_ts = self.actor_net((s1_ts1, s1_ts2))
            np_ts = torch.softmax(new_probs_ts, dim=1)

            entropy_loss = Categorical(np_ts).entropy().mean()
            op_ts = op_ts.gather(1, a_ts)
            np_ts = np_ts.gather(1, a_ts)

            ratio = torch.exp(torch.log(np_ts) - torch.log(op_ts))
            surrogate_loss = ratio * adv_ts

            ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            clipped_loss = ratio * adv_ts

            actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
            actor_loss -= ENTROPY_WEIGHT * entropy_loss

            if episode_id > 0:
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

            actor_loss_lst.append(actor_loss.item())
            critic_loss_lst.append(critic_loss.item())
            entropy_loss_lst.append(entropy_loss.item())
            if (batch_id + 1) % 3 == 0:
                print(
                    f"critic loss: {sum(critic_loss_lst)/len(critic_loss_lst):.8f}",
                    f"\t\tactor loss: {sum(actor_loss_lst)/len(actor_loss_lst):.8f}",
                    f"\t\tentropy loss: {sum(entropy_loss_lst)/len(entropy_loss_lst):.8f}",
                )
                actor_loss_lst.clear()
                critic_loss_lst.clear()
                entropy_loss_lst.clear()

    def get_gae(self, rewards, masks, values):
        """
        Calculate Generalized Advantage Estimation.
        """
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        previous_value = 0
        previous_return = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_tderror = (
                rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
            )
            running_advants = (
                running_tderror + GAMMA * LAMBDA * running_advants * masks[t]
            )

            returns[t] = rewards[t] + GAMMA * previous_return * masks[t]
            previous_value = values.data[t]
            previous_return = returns.data[t]
            advants[t] = running_advants
        # advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def td_error(self, rewards, masks, values):
        """
        Calculate 5-step TD Error.
        """
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        for start_t in range(0, len(rewards)):
            end_t = min(start_t + 5, len(rewards) - 1)
            approx_q_value = 0
            gamma_exponent = 0
            for t in range(start_t, end_t):
                if masks[t] == 0.0:
                    break
                approx_q_value += rewards[t] * (GAMMA**gamma_exponent)
                gamma_exponent += 1
            approx_q_value += values[end_t]
            returns[start_t] = approx_q_value
            advants[start_t] = approx_q_value - values[start_t]
        return returns, advants

    def state_preprocessor(self, s, agent_id):

        gamestate, previous_states = s

        state1 = np.zeros((self.s_dim * (PRE_STATES_NUM + 1),), dtype=np.float32)

        for state_id in range(PRE_STATES_NUM + 1):
            if state_id == 0:
                p1 = gamestate.players[agent_id]
                p2 = gamestate.players[3 - agent_id]
            else:
                p1 = previous_states[state_id - 1].players[agent_id]
                p2 = previous_states[state_id - 1].players[3 - agent_id]

            state1[self.s_dim * state_id + 0] = p1.position.x / 90
            state1[self.s_dim * state_id + 1] = p1.position.y / 90
            state1[self.s_dim * state_id + 2] = p2.position.x / 90
            state1[self.s_dim * state_id + 3] = p2.position.y / 90
            state1[self.s_dim * state_id + 4] = (p1.position.x - p2.position.x) / 45
            state1[self.s_dim * state_id + 5] = (p1.position.y - p2.position.y) / 45
            state1[self.s_dim * state_id + 6] = 1.0 if p1.facing else -1.0
            state1[self.s_dim * state_id + 7] = 1.0 if p2.facing else -1.0
            state1[self.s_dim * state_id + 8] = (
                1.0 if (p1.position.x - p2.position.x) * state1[6] < 0 else -1.0
            )
            state1[self.s_dim * state_id + 9] = log(
                abs(p1.position.x - p2.position.x) + 1
            )
            state1[self.s_dim * state_id + 10] = log(
                abs(p1.position.y - p2.position.y) + 1
            )
            state1[self.s_dim * state_id + 11] = p1.hitstun_frames_left / 10
            state1[self.s_dim * state_id + 12] = p2.hitstun_frames_left / 10
            state1[self.s_dim * state_id + 13] = p1.invulnerability_left / 20
            state1[self.s_dim * state_id + 14] = p2.invulnerability_left / 20
            state1[self.s_dim * state_id + 15] = p1.jumps_left - 1
            state1[self.s_dim * state_id + 16] = p2.jumps_left - 1
            state1[self.s_dim * state_id + 17] = p1.off_stage * 1.0
            state1[self.s_dim * state_id + 18] = p2.off_stage * 1.0
            state1[self.s_dim * state_id + 19] = p1.on_ground * 1.0
            state1[self.s_dim * state_id + 20] = p2.on_ground * 1.0
            state1[self.s_dim * state_id + 21] = (p1.percent - 50) / 50
            state1[self.s_dim * state_id + 22] = (p2.percent - 50) / 50
            state1[self.s_dim * state_id + 23] = (p1.shield_strength - 30) / 30
            state1[self.s_dim * state_id + 24] = (p2.shield_strength - 30) / 30
            state1[self.s_dim * state_id + 25] = p1.speed_air_x_self
            state1[self.s_dim * state_id + 26] = p2.speed_air_x_self
            state1[self.s_dim * state_id + 27] = p1.speed_ground_x_self
            state1[self.s_dim * state_id + 28] = p2.speed_ground_x_self
            state1[self.s_dim * state_id + 29] = p1.speed_x_attack
            state1[self.s_dim * state_id + 30] = p2.speed_x_attack
            state1[self.s_dim * state_id + 31] = p1.speed_y_attack
            state1[self.s_dim * state_id + 32] = p2.speed_y_attack
            state1[self.s_dim * state_id + 33] = p1.speed_y_self
            state1[self.s_dim * state_id + 34] = p2.speed_y_self
            state1[self.s_dim * state_id + 35] = (p1.action_frame - 15) / 15
            state1[self.s_dim * state_id + 36] = (p2.action_frame - 15) / 15
            if p1.action.value < 386:
                state1[self.s_dim * state_id + 37 + p1.action.value] = 1.0
            if p2.action.value < 386:
                state1[self.s_dim * state_id + 37 + 386 + p2.action.value] = 1.0

        state2 = np.zeros((1,), dtype=np.float32)  # not using this one yet

        return (state1, state2)
