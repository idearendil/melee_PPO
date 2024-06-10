"""
ReplayBuffer class file.
"""

from collections import deque
import random
import numpy as np
import torch
from parameters import EPISODE_LEN, PREDICTION_NUM


class ReplayBuffer:
    """
    Class of replay buffer.
    This replay buffer includes functions such as push and pull.
    Each data = state, action, advant, return, action_prob.
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.pickable_lst = []

    def push(self, data):
        """
        Push one set of data into buffer.

        :arg data:
            data should be a tuple of \
                s1, s2, a, op, adv, ret, \
                    actor_hs, actor_cs, critic_hs, critic_cs, pickable.
        """

        if data[-1]:
            self.pickable_lst.append(len(self.buffer))
        self.buffer.append(data[:-1])

    def pull(self, data_size, observation_normalizer, device="cpu"):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :arg device:    
            The device where to convey the data.

        :return:
            A tuple which consists of tensors of \
                s1, s2, a, op, adv, ret, \
                    actor_hs, actor_cs, critic_hs, critic_cs.
        """

        start_ids = random.sample(self.pickable_lst, data_size)

        s1_lst, s2_lst = [], []
        a_lst, op_lst, adv_lst, ret_lst = [], [], [], []
        actor_hs_lst, actor_cs_lst, critic_hs_lst, critic_cs_lst = [], [], [], []

        for data_id in range(data_size):
            s1_lst_t, s2_lst_t = [], []
            a_lst_t, op_lst_t, adv_lst_t, ret_lst_t = [], [], [], []

            start_id = start_ids[data_id]
            end_id = start_ids[data_id] + EPISODE_LEN
            for idx in range(start_id, end_id):
                s1_lst_t.append(self.buffer[idx][0])
                s2_lst_t.append(self.buffer[idx][1])
                if idx >= end_id - PREDICTION_NUM:
                    a_lst_t.append(self.buffer[idx][2])
                    op_lst_t.append(self.buffer[idx][3])
                    adv_lst_t.append(self.buffer[idx][4])
                    ret_lst_t.append(self.buffer[idx][5])

            s1_lst.append(np.stack(s1_lst_t, axis=0))
            s2_lst.append(np.stack(s2_lst_t, axis=0))

            a_lst.append(np.array(a_lst_t))
            op_lst.append(torch.stack(op_lst_t, dim=0))
            adv_lst.append(np.array(adv_lst_t))
            ret_lst.append(np.array(ret_lst_t))

            actor_hs_lst.append(self.buffer[start_id][6])
            actor_cs_lst.append(self.buffer[start_id][7])
            critic_hs_lst.append(self.buffer[start_id][8])
            critic_cs_lst.append(self.buffer[start_id][9])

        s1_np = np.stack(s1_lst, axis=0)
        s2_np = np.stack(s2_lst, axis=0)
        s1_np, s2_np = observation_normalizer(s1_np, s2_np)
        s1_ts = torch.Tensor(s1_np)
        s2_ts = torch.Tensor(s2_np)

        a_ts = torch.LongTensor(np.concatenate(a_lst, axis=0)).unsqueeze(1)
        op_ts = torch.cat(op_lst, dim=0)
        adv_ts = torch.Tensor(np.concatenate(adv_lst, axis=0)).unsqueeze(1)
        ret_ts = torch.Tensor(np.concatenate(ret_lst, axis=0)).unsqueeze(1)

        actor_hs = torch.cat(actor_hs_lst, dim=1)
        actor_cs = torch.cat(actor_cs_lst, dim=1)
        critic_hs = torch.cat(critic_hs_lst, dim=1)
        critic_cs = torch.cat(critic_cs_lst, dim=1)

        return (
            s1_ts.to(device),
            s2_ts.to(device),
            a_ts.to(device),
            op_ts.to(device),
            adv_ts.to(device),
            ret_ts.to(device),
            actor_hs.to(device),
            actor_cs.to(device),
            critic_hs.to(device),
            critic_cs.to(device),
        )

    def size(self):
        """
        Returns the size of buffer.
        """
        return len(self.buffer)

    def clear(self):
        """
        Clear buffer.
        """
        self.buffer.clear()
        self.pickable_lst.clear()
