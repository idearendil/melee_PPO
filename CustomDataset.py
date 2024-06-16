import torch
import numpy as np

class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self, buffer, pickable_lst, episode_len, device):
        self.buffer = buffer
        self.pickable_lst = pickable_lst
        self.episode_len = episode_len
        self.device = device

    def __len__(self):
        return len(self.pickable_lst)

    def __getitem__(self, idx):
        s1_lst_t, s2_lst_t = [], []
        a_lst_t, op_lst_t, adv_lst_t, ret_lst_t = [], [], [], []

        start_id = self.pickable_lst[idx]
        end_id = start_id + EPISODE_LEN
        for idx in range(start_id, end_id):
            s1_lst_t.append(self.buffer[idx][0])
            s2_lst_t.append(self.buffer[idx][1])
            if idx >= end_id - PREDICTION_NUM:
                a_lst_t.append(self.buffer[idx][2])
                op_lst_t.append(self.buffer[idx][3])
                adv_lst_t.append(self.buffer[idx][4])
                ret_lst_t.append(self.buffer[idx][5])

        s1_ts = torch.Tensors(np.stack(s1_lst_t, axis=0))
        s2_ts = torch.Tensors(np.stack(s2_lst_t, axis=0))

        a_ts = torch.LongTensor(np.array(a_lst_t)).unsqueeze(1)
        op_ts = torch.stack(op_lst_t, dim=0)
        adv_ts = torch.Tensor(np.array(adv_lst_t)).unsqueeze(1)
        ret_ts = torch.Tensor(np.array(ret_lst_t)).unsqueeze(1)

        actor_hs = self.buffer[start_id][6]
        actor_cs = self.buffer[start_id][7]
        critic_hs = self.buffer[start_id][8]
        critic_cs = self.buffer[start_id][9]

        return (
            s1_ts.to(self.device),
            s2_ts.to(self.device),
            a_ts.to(self.device),
            op_ts.to(self.device),
            adv_ts.to(self.device),
            ret_ts.to(self.device),
            actor_hs.to(self.device),
            actor_cs.to(self.device),
            critic_hs.to(self.device),
            critic_cs.to(self.device),
        )
