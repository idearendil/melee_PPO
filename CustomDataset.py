import torch
import numpy as np

class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self, buffer, pickable_lst, episode_len, prediction_num):
        self.buffer = buffer
        self.pickable_lst = pickable_lst
        self.episode_len = episode_len
        self.prediction_num = prediction_num

    def __len__(self):
        return len(self.pickable_lst)

    def __getitem__(self, idx):
        s1_lst_t, s2_lst_t = [], []
        a_lst_t, op_lst_t, adv_lst_t, ret_lst_t = [], [], [], []

        start_id = self.pickable_lst[idx]
        end_id = start_id + self.episode_len
        for idx in range(start_id, end_id):
            s1_lst_t.append(self.buffer[idx][0])
            s2_lst_t.append(self.buffer[idx][1])
            if idx >= end_id - self.prediction_num:
                a_lst_t.append(self.buffer[idx][2])
                op_lst_t.append(self.buffer[idx][3])
                adv_lst_t.append(self.buffer[idx][4])
                ret_lst_t.append(self.buffer[idx][5])

        s1_ts = torch.Tensor(np.stack(s1_lst_t, axis=0))
        s2_ts = torch.Tensor(np.stack(s2_lst_t, axis=0))

        a_ts = torch.LongTensor(np.array(a_lst_t))
        op_ts = torch.stack(op_lst_t, dim=0)
        adv_ts = torch.Tensor(np.array(adv_lst_t))
        ret_ts = torch.Tensor(np.array(ret_lst_t))

        actor_hs = self.buffer[start_id][6]
        actor_cs = self.buffer[start_id][7]
        critic_hs = self.buffer[start_id][8]
        critic_cs = self.buffer[start_id][9]

        return (
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
        )
