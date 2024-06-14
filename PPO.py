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
    EPISODE_LEN,
    EPSILON,
    L2_RATE,
    BUFFER_SIZE,
    BATCH_NUM,
    ENTROPY_WEIGHT,
    PRE_STATES_NUM,
    PREDICTION_NUM,
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

    def choose_action(self, s, agent_id, hs_cs=None, device="cpu"):
        """
        Convert state to preprocessed tensor,
        and then return action probability by actor_net
        """
        s_np = self.state_preprocessor(s, agent_id)
        s_ts = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return self.actor_net.choose_action(s_ts, hs_cs, device)

    def push_an_episode(self, data, agent_id):
        """
        Push an episode to replay buffer.

        Args:
            data: an array of ([s1, a], [s2, r, mask])
        Returns:
            None
        """

        s1_lst, a_lst, s2_lst, r_lst, mask_lst = [], [], [], [], []
        for data1, data2 in data:
            s1, a = data1
            s2, r, mask = data2
            s1_lst.append(s1)
            a_lst.append(a)
            s2_lst.append(s2)
            r_lst.append(r)
            mask_lst.append(mask)

        if len(s1_lst) != len(s2_lst):
            print("Error occured!!! len(s1_lst) != len(s2_lst)")

        s1_np_lst, s2_np_lst = [], []
        for s1 in s1_lst:
            s1 = self.state_preprocessor(s1, agent_id)
            s1_np_lst.append(s1)
        for s2 in s2_lst:
            s2 = self.state_preprocessor(s2, agent_id)
            s2_np_lst.append(s2)

        s1_ts = torch.Tensor(np.array(s1_np_lst, dtype=np.float32)).to(self.device)
        s2_ts = torch.Tensor(np.array(s2_np_lst, dtype=np.float32)).to(self.device)
        r_ts = torch.Tensor(np.array(r_lst, dtype=np.float32))
        masks = torch.Tensor(np.array(mask_lst, dtype=np.float32))

        v_lst = []
        prob_lst = []
        actor_hs_lst = []
        actor_cs_lst = []
        critic_hs_lst = []
        critic_cs_lst = []
        with torch.no_grad():
            self.critic_net.eval()
            self.actor_net.eval()
            actor_hs_cs = (
                torch.zeros((2, 1, 256), dtype=torch.float32).to(self.device),
                torch.zeros((2, 1, 256), dtype=torch.float32).to(self.device),
            )
            critic_hs_cs = (
                torch.zeros((2, 1, 256), dtype=torch.float32).to(self.device),
                torch.zeros((2, 1, 256), dtype=torch.float32).to(self.device),
            )
            for idx, _ in enumerate(s1_lst):
                actor_hs_lst.append(actor_hs_cs[0].cpu().clone().detach())
                actor_cs_lst.append(actor_hs_cs[1].cpu().clone().detach())
                critic_hs_lst.append(critic_hs_cs[0].cpu().clone().detach())
                critic_cs_lst.append(critic_hs_cs[1].cpu().clone().detach())
                prob, actor_hs_cs = self.actor_net(
                    s1_ts[idx].unsqueeze(0).unsqueeze(0),
                    actor_hs_cs,
                )
                v, critic_hs_cs = self.critic_net(
                    s2_ts[idx].unsqueeze(0).unsqueeze(0),
                    critic_hs_cs,
                )
                prob = torch.softmax(prob.squeeze().cpu(), dim=0)
                v = v.squeeze().item()
                v_lst.append(v)
                prob_lst.append(prob)
            v_ts = torch.FloatTensor(v_lst)
            ret_ts, adv_ts = self.td_error(r_ts, masks, v_ts)

        for idx, _ in enumerate(s1_lst):
            self.buffer.push(
                [
                    s1_np_lst[idx],
                    s2_np_lst[idx],
                    a_lst[idx],
                    prob_lst[idx],
                    adv_ts[idx].item(),
                    ret_ts[idx].item(),
                    actor_hs_lst[idx],
                    actor_cs_lst[idx],
                    critic_hs_lst[idx],
                    critic_cs_lst[idx],
                    len(s1_lst) > idx + EPISODE_LEN,
                ]
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
            (
                s1_ts,
                s2_ts,
                a_ts,
                op_ts,
                adv_ts,
                ret_ts,
                actor_hs,
                actor_cs,
                critic_hs,
                critic_cs,
            ) = self.buffer.pull(BATCH_SIZE, self.device)

            v_ts, _ = self.critic_net(s2_ts, (critic_hs, critic_cs))
            v_ts = v_ts[:, -PREDICTION_NUM:, :]
            v_ts = v_ts.reshape((-1, 1))
            critic_loss = self.critic_loss_func(v_ts, ret_ts)
            if episode_id > 0:
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            new_probs_ts, _ = self.actor_net(s1_ts, (actor_hs, actor_cs))
            new_probs_ts = new_probs_ts[:, -PREDICTION_NUM:, :]
            new_probs_ts = new_probs_ts.reshape((-1, op_ts.shape[1]))
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

            if episode_id > 200:
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

            actor_loss_lst.append(actor_loss.item())
            critic_loss_lst.append(critic_loss.item())
            entropy_loss_lst.append(entropy_loss.item())
            if (batch_id + 1) % 10 == 0:
                critic_loss_avg = sum(critic_loss_lst) / len(critic_loss_lst)
                actor_loss_avg = sum(actor_loss_lst) / len(actor_loss_lst)
                entropy_loss_avg = sum(entropy_loss_lst) / len(entropy_loss_lst)
                if episode_id <= 0:
                    critic_loss_avg = "not trained yet!"
                if episode_id <= 200:
                    actor_loss_avg = "not trained yet!"
                    entropy_loss_avg = "not trained yet!"
                print(
                    f"critic loss: {critic_loss_avg}",
                    f"\tactor loss: {actor_loss_avg}",
                    f"\tentropy loss: {entropy_loss_avg}",
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
                if t == end_t - 1:
                    approx_q_value += values[end_t] * (GAMMA**gamma_exponent)
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
                1.0
                if (p1.position.x - p2.position.x) * state1[self.s_dim * state_id + 6]
                < 0
                else -1.0
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
            state1[self.s_dim * state_id + 25] = p1.speed_air_x_self / 2
            state1[self.s_dim * state_id + 26] = p2.speed_air_x_self / 2
            state1[self.s_dim * state_id + 27] = p1.speed_ground_x_self / 2
            state1[self.s_dim * state_id + 28] = p2.speed_ground_x_self / 2
            state1[self.s_dim * state_id + 29] = p1.speed_x_attack
            state1[self.s_dim * state_id + 30] = p2.speed_x_attack
            state1[self.s_dim * state_id + 31] = p1.speed_y_attack
            state1[self.s_dim * state_id + 32] = p2.speed_y_attack
            state1[self.s_dim * state_id + 33] = p1.speed_y_self
            state1[self.s_dim * state_id + 34] = p2.speed_y_self
            state1[self.s_dim * state_id + 35] = (p1.action_frame - 15) / 15
            state1[self.s_dim * state_id + 36] = (p2.action_frame - 15) / 15
            state1[self.s_dim * state_id + 37] = p1.stock - 2.5
            state1[self.s_dim * state_id + 38] = p2.stock - 2.5
            if p1.action.value < 386:
                state1[self.s_dim * state_id + 39 + p1.action.value] = 1.0
            if p2.action.value < 386:
                state1[self.s_dim * state_id + 39 + 386 + p2.action.value] = 1.0

        return state1
