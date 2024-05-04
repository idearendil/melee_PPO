"""
Start training models by PPO method within melee environment.
"""

import argparse
import random
import torch
from pynput.keyboard import Key, Controller
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from melee import enums
from parameters import (
    MAX_STEP,
    CYCLE_NUM,
    MIN_TUPLES_IN_CYCLE,
    STATE_DIM,
    ACTION_DIM,
    DELAY,
)

# from observation_normalizer import ObservationNormalizer
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP, CPU


parser = argparse.ArgumentParser()
parser.add_argument(
    "--continue_training",
    type=bool,
    default=False,
    help="whether to continue training with existing models",
)
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
        NOOP(enums.Character.FOX),
    ]

    episode_id = 0
    if args.continue_training:
        # load pre-trained models
        print("continue training from the latest models...")
        players[0].ppo.actor_net = torch.load(args.model_path + "actor_net.pt").to(
            device
        )
        players[0].ppo.critic_net = torch.load(args.model_path + "critic_net.pt").to(
            device
        )
        df = pd.read_csv("log_train.csv")
        episode_id = len(df) - 1
    else:
        # clear log file
        with open("log_train.csv", "w", encoding="utf-8") as outfile:
            outfile.write("episode_id,score\n")

    for cycle_id in range(CYCLE_NUM):
        scores = []  # for log
        players[0].ppo.buffer.buffer.clear()  # PPO is an on-policy algorithm
        while players[0].ppo.buffer.size() < MIN_TUPLES_IN_CYCLE:
            episode_id += 1
            score = 0
            fucked_up_cnt = 0

            players[1] = CPU(enums.Character.FOX, random.randint(1, 9))

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
                        episode_buffer.append(
                            [now_s, act_data[0], act_data[1], step_cnt]
                        )
                    action_pair[0] = now_action
                    action_pair[1] = players[1].act(now_s[0])

                    now_s, r, done, _ = env.step(*action_pair)
                    mask = (1 - done) * 1
                    score += r[0]  # for log

                    r_sum += r[0]
                    mask_sum *= mask

                    if done:
                        # if finished, add last information to episode memory
                        temp = episode_buffer[last_state_idx]
                        episode_memory.append(
                            [temp[0], temp[1], r_sum, mask_sum, temp[2]]
                        )
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
                "episode:",
                episode_id,
                "\tbuffer length:",
                players[0].ppo.buffer.size(),
                "\tfucked up:",
                fucked_up_cnt,
            )

            with open("log_train.csv", "a", encoding="utf-8") as outfile:
                outfile.write(str(episode_id) + "," + str(score) + "\n")
            scores.append(score)

        print("cycle: ", cycle_id, "\tscore: ", np.mean(scores))

        players[0].ppo.train()
        torch.save(players[0].ppo.actor_net, args.model_path + "actor_net.pt")
        torch.save(players[0].ppo.critic_net, args.model_path + "critic_net.pt")

    log_df = pd.read_csv("log_train.csv")
    plt.plot(log_df["episode_id"], log_df["score"])
    plt.show()


if __name__ == "__main__":
    run()
