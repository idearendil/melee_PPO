"""
ReplayBuffer class file.
"""

from collections import deque
import random


class ReplayBuffer:
    """
    Class of replay buffer.
    This replay buffer includes functions such as push and pull.
    Each data = state, action, advant, return, action_prob.
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, data):
        """
        Push one set of data into buffer.

        :arg data:
            data should be a tuple of \
                state, action, advant, return, action_prob.
        """

        self.buffer.append(data)

    def pull(self, data_size):
        """
        Pull data of size data_size from buffer.

        :arg data_size:
            The size of data which will be pulled from buffer.

        :return:
            A tuple which consists of lists of \
                state1, state2, action, advant, return, action_prob, \
                    actor_hs_cs, critic_hs_cs, agent_id.
        """
        minibatch = random.sample(self.buffer, data_size)
        s1_lst, s2_lst, a_lst, adv_lst, ret_lst, op_lst = [], [], [], [], [], []
        actor_hs_cs_lst, critic_hs_cs_lst, id_lst = [], [], []

        for an_episode in minibatch:
            s1_lst.append(an_episode[0])
            s2_lst.append(an_episode[1])
            a_lst.extend(an_episode[2])
            adv_lst.append(an_episode[3])
            ret_lst.append(an_episode[4])
            op_lst.append(an_episode[5])
            actor_hs_cs_lst.append(an_episode[6])
            critic_hs_cs_lst.append(an_episode[7])
            id_lst.append(an_episode[8])

        return (
            s1_lst,
            s2_lst,
            a_lst,
            adv_lst,
            ret_lst,
            op_lst,
            actor_hs_cs_lst,
            critic_hs_cs_lst,
            id_lst,
        )

    def size(self):
        """
        Returns the size of buffer.
        """
        return len(self.buffer)
