"""
Start training models by PPO method within melee environment.
"""

import argparse
import torch
from pynput.keyboard import Key, Controller
import numpy as np
from melee import enums
from parameters import (
    MAX_STEP,
    STATE_DIM,
    ACTION_DIM,
    DELAY,
)
import pickle

from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="./models_train/", help="where models are saved"
)
parser.add_argument(
    "--iso",
    type=str,
    default="../ssbm.iso",
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO",
)
parser.add_argument(
    "--fastforward",
    type=int,
    default=1,
    help="whether to turn up fastforward",
)
parser.add_argument(
    "--episode_num", type=int, default=5000, help="How many times to test"
)
args = parser.parse_args()


def run():
    """
    Start training with given options.
    """

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(device)

    torch.manual_seed(500)
    np.random.seed(500)

    players = [
        PPOAgent(enums.Character.FOX, 1, 2, device, STATE_DIM, ACTION_DIM),
        PPOAgent(enums.Character.FOX, 2, 1, device, STATE_DIM, ACTION_DIM),
    ]

    players[0].ppo.actor_net = torch.load(args.model_path + "actor_net_last.pt")
    players[0].ppo.critic_net = torch.load(args.model_path + "critic_net_last.pt")
    players[1].ppo.actor_net = torch.load(args.model_path + "actor_net_last.pt")
    players[1].ppo.critic_net = torch.load(args.model_path + "critic_net_last.pt")

    for episode_id in range(1936, args.episode_num):
        players[0].ppo.buffer.buffer.clear()

        score = 0
        fucked_up_cnt = 0

        env = MeleeEnv(args.iso, players, fast_forward=True)
        env.start()
        if args.fastforward:
            Controller().press(Key.tab)
        now_s, _ = env.reset(enums.Stage.FINAL_DESTINATION)

        episode_memory = []
        episode_buffer = []

        action_pair = [0, 0]

        r_sum = 0
        mask_sum = 1

        last_state_idx = -1

        players[0].ppo.actor_net.eval()
        for step_cnt in range(MAX_STEP):
            if step_cnt > 100:  # if step_cnt < 100, it's not started yet

                now_action, act_data = players[0].act(now_s)
                if act_data is not None:
                    episode_buffer.append([now_s, act_data[0], act_data[1], step_cnt])
                action_pair[0] = now_action
                action_pair[1], _ = players[1].act(now_s)

                now_s, r, done, _ = env.step(*action_pair)
                mask = (1 - done) * 1
                score += r[0]  # for log

                r_sum += r[0]
                mask_sum *= mask

                if done:
                    # if finished, add last information to episode memory
                    temp = episode_buffer[last_state_idx]
                    episode_memory.append([temp[0], temp[1], r_sum, mask_sum, temp[2]])
                    break

                if (
                    now_s[0].players[1].action_frame == 1
                    and now_s[0].players[1].position.y >= 0
                ):
                    # if agent's new action animation just started
                    p1_action = now_s[0].players[1].action
                    if p1_action in players[0].action_space.sensor:
                        # if agent's animation is in sensor set
                        # find action which caused agent's current animation
                        action_candidate = players[0].action_space.sensor[p1_action]
                        action_is_found = False
                        for i in range(len(episode_buffer) - 1, last_state_idx, -1):

                            if episode_buffer[i][3] > step_cnt - DELAY:
                                # action can cause animation after 2 frames at least
                                continue

                            if episode_buffer[i][1] in action_candidate:
                                if last_state_idx >= 0:
                                    # save last action and its consequence in episode memory
                                    temp = episode_buffer[last_state_idx]
                                    episode_memory.append(
                                        [temp[0], temp[1], r_sum, mask_sum, temp[2]]
                                    )
                                r_sum = 0
                                mask_sum = 1
                                last_state_idx = i
                                action_is_found = True
                                break

                        if not action_is_found:
                            fucked_up_cnt += 1
            else:
                action_pair = [0, 0]
                now_s, _, _, _ = env.step(*action_pair)

        env.close()
        if args.fastforward:
            Controller().release(Key.tab)

        players[0].ppo.push_an_episode(episode_memory, 1)

        print(
            "buffer length:",
            players[0].ppo.buffer.size(),
            "\tfucked up:",
            fucked_up_cnt,
        )
        print("cycle: ", episode_id, "\tscore: ", score)

        dataset = []
        for a_tuple in players[0].ppo.buffer.buffer:
            dataset.append((a_tuple[0][0], a_tuple[3]))
        with open(
            "E:/train_framework_value/dataset/dataset" + str(episode_id) + ".pickle",
            "wb",
        ) as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run()
