"""
The file of Ppo class.
"""

import torch.optim as optim
import numpy as np
import torch
from torch.distributions import Categorical
from model import Actor, Critic
from parameters import LR_ACTOR, LR_CRITIC, GAMMA, LAMBDA, BATCH_SIZE, \
    EPSILON, L2_RATE, BUFFER_SIZE, BATCH_NUM, ENTROPY_WEIGHT, DELAY
from replay_buffer import ReplayBuffer


class Ppo:
    """
    The class which Proximal Policy Optimization is implemented in.
    """
    def __init__(self, s_dim, a_dim, device):
        self.device = device
        self.actor_net = Actor(s_dim, a_dim).to(self.device)
        self.critic_net = Critic(s_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(
            self.critic_net.parameters(), lr=LR_CRITIC, weight_decay=L2_RATE
        )
        self.critic_loss_func = torch.nn.MSELoss()
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def models_to_device(self, device):
        """
        Move actor and critic to the specific device('cpu' or 'cuda:x')
        """
        self.actor_net.to(device)
        self.critic_net.to(device)

    def choose_action(self, s):
        s_ts1, s_ts2 = self.state_preprocessor(s)
        s_ts1 = torch.from_numpy(s_ts1).unsqueeze(0).to(self.device)
        s_ts2 = torch.from_numpy(s_ts2).unsqueeze(0).to(self.device)
        return self.actor_net.choose_action((s_ts1, s_ts2))

    def push_an_episode(self, data):
        """
        Push an episode to replay buffer.

        Args:
            data: an array of (state, action, reward, mask).
        Returns:
            None
        """

        s_lst1, a_lst, r_lst, mask_lst, prob_lst = \
            [], [], [], [], []
        for s, a, r, mask, prob in data:
            s_lst1.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(mask)
            prob_lst.append(torch.Tensor(prob))

        s_lst2 = []
        for s in s_lst1:
            _, s2 = self.state_preprocessor(s)
            s_lst2.append(s2)

        s_ts1 = torch.Tensor(
            np.array(s_lst1, dtype=np.float32)).to(self.device)
        s_ts2 = torch.Tensor(
            np.array(s_lst2, dtype=np.float32)).to(self.device)
        r_ts = torch.Tensor(np.array(r_lst, dtype=np.float32))
        masks = torch.Tensor(np.array(mask_lst, dtype=np.float32))

        with torch.no_grad():
            self.critic_net.eval()
            v_lst = []
            for idx in range(0, len(s_ts1), BATCH_SIZE):
                idx_end = min(idx+BATCH_SIZE, len(s_ts1))
                v_lst.append(self.critic_net((s_ts1[idx:idx_end],
                                              s_ts2[idx:idx_end])).cpu())
            v_ts = torch.concatenate(v_lst, dim=0)
            ret_ts, adv_ts = self.get_gae(r_ts, masks, v_ts)

        for idx, _ in enumerate(s_ts1):
            if idx+DELAY >= len(s_ts1):
                break
            self.buffer.push((s_lst1[idx],
                              a_lst[idx],
                              adv_ts[idx+DELAY].item(),
                              ret_ts[idx].item(),
                              prob_lst[idx]))

    def train(self):
        """
        Train Actor network and Value network with data in buffer.
        """
        print('buffer size: ', self.buffer.size())

        self.actor_net.train()
        self.critic_net.train()
        critic_loss_lst, actor_loss_lst = [], []
        for batch_id in range(BATCH_NUM):

            s_lst1, a_lst, adv_lst, ret_lst, op_lst = self.buffer.pull(
                BATCH_SIZE)

            s_lst2 = []
            for s in s_lst1:
                _, s2 = self.state_preprocessor(s)
                s_lst2.append(s2)

            st_ts1 = torch.Tensor(np.stack(s_lst1, axis=0)).to(self.device)
            st_ts2 = torch.Tensor(np.stack(s_lst2, axis=0)).to(self.device)
            a_ts = torch.LongTensor(a_lst).to(self.device).unsqueeze(1)
            adv_ts = torch.Tensor(np.array(
                adv_lst, dtype=np.float32)).unsqueeze(1).to(self.device)
            ret_ts = torch.Tensor(np.array(
                ret_lst, dtype=np.float32)).unsqueeze(1).to(self.device)
            op_ts = torch.Tensor(np.stack(op_lst, axis=0)).to(self.device)

            v_ts = self.critic_net((st_ts1, st_ts2))
            critic_loss = self.critic_loss_func(v_ts, ret_ts)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            new_probs_ts = torch.softmax(self.actor_net((st_ts1, st_ts2)),
                                         dim=1)
            entropy_loss = Categorical(new_probs_ts).entropy().mean()
            op_ts = op_ts.gather(1, a_ts)
            new_probs_ts = new_probs_ts.gather(1, a_ts)

            ratio = torch.exp(torch.log(new_probs_ts) - torch.log(op_ts))
            surrogate_loss = ratio * adv_ts

            ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            clipped_loss = ratio * adv_ts

            actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
            actor_loss -= ENTROPY_WEIGHT * entropy_loss
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            actor_loss_lst.append(actor_loss.item())            
            critic_loss_lst.append(critic_loss.item())
            if batch_id % 10 == 0:
                print(f'critic loss: {sum(critic_loss_lst)/len(critic_loss_lst):.8f}',
                      f'\t\tactor loss: {sum(actor_loss_lst)/len(actor_loss_lst):.8f}')
                actor_loss_lst.clear()
                critic_loss_lst.clear()

    def kl_divergence(self, old_mu, old_sigma, mu, sigma):
        """
        KL divergence of two different normal distributions.
        """
        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()
        kl = (
            torch.log(old_sigma)
            - torch.log(sigma)
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5
        )
        return kl.sum(1, keepdim=True)

    def get_gae(self, rewards, masks, values):
        """
        Calculate Generalized Advantage Estimation.
        """
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + GAMMA * running_returns * masks[t]
            running_tderror = (
                rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
            )
            running_advants = (
                running_tderror + GAMMA * LAMBDA * running_advants * masks[t]
            )

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def state_preprocessor(self, s):
        # size = (200, 200)
        # center point = (0, 0) => (100, 40) <= this might be modified
        def coordinator(x, y):
            return int(min(max(x + 100, 0), 199)), int(min(max(y + 40, 0), 199))

        image_tensor = np.zeros((3, 200, 200), dtype=np.float32)
        p1_x = s[0]
        p1_y = s[2]
        p2_x = s[1]
        p2_y = s[3]
        p1_face = 1.0 if s[6] else -1.0
        p2_face = 1.0 if s[7] else -1.0
        p1_x, p1_y = coordinator(p1_x, p1_y)
        p2_x, p2_y = coordinator(p2_x, p2_y)
        image_tensor[1, p1_x, p1_y] = p1_face
        image_tensor[2, p2_x, p2_y] = p2_face

        for i in range(-70, 71):
            image_tensor[0, i, 0] = 1.0
        for i in range(-60, -20):
            image_tensor[0, i, 27] = 1.0
        for i in range(21, 61):
            image_tensor[0, i, 27] = 1.0
        for i in range(-20, 21):
            image_tensor[0, i, 54] = 1.0

        return (s, image_tensor)
