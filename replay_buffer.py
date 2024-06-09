"""
ReplayBuffer class file.
"""

from collections import deque
import random
import numpy as np
import torch
from parameters import EPISODE_LEN


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
                s1_1, s1_2, s2_1, s2_2, a, op, adv, ret, \
                    actor_hs, actor_cs, critic_hs, critic_cs, pickable.
        """

        if data[-1]:
            self.pickable_lst.append(len(self.buffer))
        self.buffer.append(data[:-1])

    def pull(self, data_size, device="cpu"):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :arg device:
            The device where to convey the data.

        :return:
            A tuple which consists of tensors of \
                s1_1, s1_2, s2_1, s2_2, a, op, adv, ret, \
                    actor_hs, actor_cs, critic_hs, critic_cs.
        """

        start_ids = random.sample(self.pickable_lst, data_size)

        s1_lst1, s1_lst2, s2_lst1, s2_lst2 = [], [], [], []
        a_lst, op_lst, adv_lst, ret_lst = [], [], [], []
        actor_hs_lst, actor_cs_lst, critic_hs_lst, critic_cs_lst = [], [], [], []
        for data_id in range(data_size):
            s1_lst1_t, s1_lst2_t, s2_lst1_t, s2_lst2_t = [], [], [], []
            a_lst_t, op_lst_t, adv_lst_t, ret_lst_t = [], [], [], []
            start_id = start_ids[data_id]
            end_id = start_ids[data_id] + EPISODE_LEN
            for idx in range(start_id, end_id):
                s1_lst1_t.append(self.buffer[idx][0])
                s1_lst2_t.append(self.buffer[idx][1])
                s2_lst1_t.append(self.buffer[idx][2])
                s2_lst2_t.append(self.buffer[idx][3])
                a_lst_t.append(self.buffer[idx][4])
                op_lst_t.append(self.buffer[idx][5])
                adv_lst_t.append(self.buffer[idx][6])
                ret_lst_t.append(self.buffer[idx][7])
            s1_lst1.append(np.stack(s1_lst1_t, axis=0))
            s1_lst2.append(np.stack(s1_lst2_t, axis=0))
            s2_lst1.append(np.stack(s2_lst1_t, axis=0))
            s2_lst2.append(np.stack(s2_lst2_t, axis=0))
            a_lst.append(np.array(a_lst_t))
            op_lst.append(torch.stack(op_lst_t, dim=0))
            adv_lst.append(np.array(adv_lst_t))
            ret_lst.append(np.array(ret_lst_t))
            actor_hs_lst.append(self.buffer[start_id][8])
            actor_cs_lst.append(self.buffer[start_id][9])
            critic_hs_lst.append(self.buffer[start_id][10])
            critic_cs_lst.append(self.buffer[start_id][11])
        s1_ts1 = torch.Tensor(np.stack(s1_lst1, axis=0))
        s1_ts2 = torch.Tensor(np.stack(s1_lst2, axis=0))
        s2_ts1 = torch.Tensor(np.stack(s2_lst1, axis=0))
        s2_ts2 = torch.Tensor(np.stack(s2_lst2, axis=0))
        a_ts = torch.LongTensor(np.concatenate(a_lst, axis=0)).unsqueeze(1)
        op_ts = torch.cat(op_lst, dim=0)
        adv_ts = torch.Tensor(np.concatenate(adv_lst, axis=0)).unsqueeze(1)
        ret_ts = torch.Tensor(np.concatenate(ret_lst, axis=0)).unsqueeze(1)
        actor_hs = torch.cat(actor_hs_lst, dim=1)
        actor_cs = torch.cat(actor_cs_lst, dim=1)
        critic_hs = torch.cat(critic_hs_lst, dim=1)
        critic_cs = torch.cat(critic_cs_lst, dim=1)

        return (
            s1_ts1.to(device),
            s1_ts2.to(device),
            s2_ts1.to(device),
            s2_ts2.to(device),
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
