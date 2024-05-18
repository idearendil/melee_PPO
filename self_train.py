"""
Start self training models by PPO method within melee environment
(with league system).
"""

import argparse
import pickle
import os
import random
import torch
from pynput.keyboard import Key, Controller
from math import log
import numpy as np
import pandas as pd
from melee import enums
from parameters import (
    MAX_STEP,
    CYCLE_NUM,
    MIN_TUPLES_IN_CYCLE,
    STATE_DIM,
    ACTION_DIM,
    DELAY,
    WIN_RATE_DECAY,
)

# from observation_normalizer import ObservationNormalizer
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP, CPU


parser = argparse.ArgumentParser()
parser.add_argument(
    "--carry_on",
    type=bool,
    default=False,
    help="whether to continue training with existing models",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./models_self_train/",
    help="where models are saved",
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


def pick_opponent(league_win_rate, device):
    """
    pick an opponent with win rate of each of recorded agents
    """
    pick_prob = []
    for a_tuple in league_win_rate:
        test_num = sum(a_tuple) + 1
        lose_rate = (a_tuple[1] + 1) / test_num
        pick_prob.append(lose_rate / (log(test_num) + 1))
    pick_prob = np.array(pick_prob, dtype=np.float32)
    pick_prob = pick_prob / np.sum(pick_prob)

    # print info
    print("\t[win rate]\t\t [pick prob]")
    if sum(league_win_rate[0]) <= 0.0:
        print("CPU 7:\t", "?", "\t\t", pick_prob[0])
    else:
        print(
            f"CPU 7:\t{(league_win_rate[0][0] / sum(league_win_rate[0])):.6f}",
            "\t\t",
            pick_prob[0],
        )
    if sum(league_win_rate[1]) <= 0.0:
        print("CPU 8:\t", "?", "\t\t", pick_prob[1])
    else:
        print(
            f"CPU 8:\t{(league_win_rate[1][0] / sum(league_win_rate[1])):.6f}",
            "\t\t",
            pick_prob[1],
        )
    if sum(league_win_rate[2]) <= 0.0:
        print("CPU 9:\t", "?", "\t\t", pick_prob[2])
    else:
        print(
            f"CPU 9:\t{(league_win_rate[2][0] / sum(league_win_rate[2])):.6f}",
            "\t\t",
            pick_prob[2],
        )
    for i in range(3, len(league_win_rate)):
        if sum(league_win_rate[i]) <= 0.0:
            print(f"Agent{i - 3}:", "?       ", "\t\t", pick_prob[i])
        else:
            print(
                f"Agent{i-3}:",
                f"{(league_win_rate[i][0] / sum(league_win_rate[i])):.6f}",
                "\t\t",
                pick_prob[i],
            )

    # pick opponent
    opp_id = random.choices(list(range(len(league_win_rate))), weights=pick_prob, k=1)[
        0
    ]

    if opp_id < 3:
        return opp_id, CPU(enums.Character.FOX, 7 + opp_id)

    opp = PPOAgent(
        enums.Character.FOX, 2, 1, device, STATE_DIM, ACTION_DIM, test_mode=True
    )
    opp.ppo.actor_net = torch.load(
        args.model_path + "actor_net_" + str(opp_id - 3) + ".pt"
    ).to(device)
    opp.ppo.critic_net = torch.load(
        args.model_path + "critic_net_" + str(opp_id - 3) + ".pt"
    ).to(device)
    return opp_id, opp


def read_files(player, device):
    """
    read win_rate.pickle file or create one if there isn't,
    read latest agent's model files.
    """

    # check whether there's a win_rate file
    if "win_rate.pickle" not in os.listdir(args.model_path):
        model_num = len(os.listdir(args.model_path)) // 2
        league_win_rate = [[0, 0] for _ in range(3 + model_num)]
        with open(args.model_path + "win_rate.pickle", "wb") as f:
            pickle.dump(league_win_rate, f, pickle.HIGHEST_PROTOCOL)

    with open(args.model_path + "win_rate.pickle", "rb") as f:
        league_win_rate = pickle.load(f)

    # check whether the number of model network files matches win_rate file
    if (len(os.listdir(args.model_path)) - 3) // 2 != len(league_win_rate) - 3:
        print("error!: The number of model files does not match with win_rate file!!")
        return

    # check whether latest model network exists
    if "actor_net_last.pt" not in os.listdir(
        args.model_path
    ) or "critic_net_last.pt" not in os.listdir(args.model_path):
        print("error!: There is no latest model!!")
        return

    player.ppo.actor_net = torch.load(args.model_path + "actor_net_last.pt").to(device)
    player.ppo.critic_net = torch.load(args.model_path + "critic_net_last.pt").to(
        device
    )

    return league_win_rate


def save_files(player, league_win_rate):
    """
    save win_rate.pickle file and latest agent's model files.
    """

    with open(args.model_path + "win_rate.pickle", "wb") as f:
        pickle.dump(league_win_rate, f, pickle.HIGHEST_PROTOCOL)

    torch.save(
        player.ppo.actor_net,
        args.model_path + "actor_net_last.pt",
    )
    torch.save(
        player.ppo.critic_net,
        args.model_path + "critic_net_last.pt",
    )


def agent_release(player, new_agent_id):
    """
    save latest agent's model files.
    """
    torch.save(
        player.ppo.actor_net,
        args.model_path + "actor_net_" + str(new_agent_id) + ".pt",
    )
    torch.save(
        player.ppo.critic_net,
        args.model_path + "critic_net_" + str(new_agent_id) + ".pt",
    )


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
    league_win_rate = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    if args.carry_on:
        print("continue training from the latest models...")
        # load pre-trained models
        league_win_rate = read_files(players[0], device)
        # set episode_id as the last episode_id in log file
        episode_id = len(pd.read_csv("log_self_train.csv")) - 1
    else:
        # clear log file
        with open("log_self_train.csv", "w", encoding="utf-8") as outfile:
            outfile.write("episode_id,score\n")

    for cycle_id in range(CYCLE_NUM):
        scores = []  # for log
        players[0].ppo.buffer.buffer.clear()  # PPO is an on-policy algorithm
        while players[0].ppo.buffer.size() < MIN_TUPLES_IN_CYCLE:
            episode_id += 1
            score = 0

            opp_id, players[1] = pick_opponent(league_win_rate, device)

            env = MeleeEnv(args.iso, players, fast_forward=True)
            env.start()
            if args.fastforward:
                Controller().press(Key.tab)
            now_s, _ = env.reset(enums.Stage.FINAL_DESTINATION)

            episode_memory = []
            episode_buffer = []

            r_sum = 0
            mask_sum = 1
            last_state_idx = -1
            fucked_up_cnt = 0
            action_pair = [0, 0]

            for step_cnt in range(MAX_STEP):
                if step_cnt > 100:  # if step_cnt < 100, it's not started yet

                    # pick player1's action
                    action_pair[0], act_data = players[0].act(now_s)
                    if act_data is not None:
                        episode_buffer.append(
                            [now_s, act_data[0], act_data[1], step_cnt]
                        )

                    # pick player2's action
                    if opp_id < 3:
                        action_pair[1] = players[1].act(now_s[0])
                    else:
                        action_pair[1], _ = players[1].act(now_s)

                    now_s, r, done, _ = env.step(*action_pair)
                    mask = (1 - done) * 1
                    score += r[0]  # just for log

                    r_sum += r[0]
                    mask_sum *= mask

                    if done or step_cnt >= MAX_STEP - 1:
                        # if finished, add last information to episode memory
                        temp = episode_buffer[last_state_idx]
                        episode_memory.append([temp[0], temp[1], r_sum, 0, temp[2]])

                        # so... who won?
                        if now_s[0].players[1].stock > now_s[0].players[2].stock:
                            league_win_rate[opp_id][0] += 1.0
                        else:
                            league_win_rate[opp_id][1] += 1.0
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

            with open("log_self_train.csv", "a", encoding="utf-8") as outfile:
                outfile.write(str(episode_id) + "," + str(score) + "\n")
            scores.append(score)

        # check if the win rate satisfies agent-releasing condition.
        agent_release_flag = True
        for a_win_rate in league_win_rate:
            if a_win_rate[0] / sum(a_win_rate) < 0.6:
                agent_release_flag = False
            a_win_rate[0] *= WIN_RATE_DECAY
            a_win_rate[1] *= WIN_RATE_DECAY
        if agent_release_flag:
            league_win_rate.append([0, 0])
            agent_release(players[0], len(league_win_rate) - 4)

        print("cycle: ", cycle_id, "\tscore: ", np.mean(scores))
        players[0].ppo.train()

        save_files(players[0], league_win_rate)


if __name__ == "__main__":
    run()
