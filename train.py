"""
Start training models by PPO method within melee environment.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from melee import enums
from parameters import \
    MAX_STEP, CYCLE_NUM, MIN_TUPLES_IN_CYCLE, STATE_DIM, ACTION_DIM, DELAY
# from observation_normalizer import ObservationNormalizer
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP, CPU
import random


useless_animations = [
    enums.Action.GRAB_PULL,
    enums.Action.MARTH_COUNTER,
    enums.Action.SHINE_TURN,
    enums.Action.FTILT_HIGH,
    enums.Action.STANDING,
    enums.Action.KNEE_BEND,
    enums.Action.ENTRY,
    enums.Action.ENTRY_START,
    enums.Action.ENTRY_END,
    enums.Action.FALLING,
    enums.Action.LANDING,
    enums.Action.DEAD_FALL,
    enums.Action.DEAD_DOWN,
    enums.Action.ON_HALO_DESCENT,
    enums.Action.ON_HALO_WAIT,
    enums.Action.GRABBED,
    enums.Action.GRAB_PUMMELED,
    enums.Action.CROUCH_START,
    enums.Action.TECH_MISS_UP,
    enums.Action.TECH_MISS_DOWN,
    enums.Action.WALK_SLOW,
    enums.Action.CROUCHING,
    enums.Action.THROWN_BACK,
    enums.Action.THROWN_COPY_STAR,
    enums.Action.THROWN_CRAZY_HAND,
    enums.Action.THROWN_DOWN,
    enums.Action.THROWN_DOWN_2,
    enums.Action.THROWN_FB,
    enums.Action.THROWN_FF,
    enums.Action.THROWN_FORWARD,
    enums.Action.THROWN_F_HIGH,
    enums.Action.THROWN_F_LOW,
    enums.Action.THROWN_UP,
    enums.Action.DAMAGE_AIR_1,
    enums.Action.DAMAGE_AIR_2,
    enums.Action.DAMAGE_AIR_3,
    enums.Action.DAMAGE_BIND,
    enums.Action.DAMAGE_FLY_HIGH,
    enums.Action.DAMAGE_FLY_LOW,
    enums.Action.DAMAGE_FLY_NEUTRAL,
    enums.Action.DAMAGE_FLY_ROLL,
    enums.Action.DAMAGE_FLY_TOP,
    enums.Action.DAMAGE_GROUND,
    enums.Action.DAMAGE_HIGH_1,
    enums.Action.DAMAGE_HIGH_2,
    enums.Action.DAMAGE_HIGH_3,
    enums.Action.DAMAGE_ICE,
    enums.Action.DAMAGE_ICE_JUMP,
    enums.Action.DAMAGE_LOW_1,
    enums.Action.DAMAGE_LOW_2,
    enums.Action.DAMAGE_LOW_3,
    enums.Action.DAMAGE_NEUTRAL_1,
    enums.Action.DAMAGE_NEUTRAL_2,
    enums.Action.DAMAGE_NEUTRAL_3,
    enums.Action.DAMAGE_SCREW,
    enums.Action.DAMAGE_SCREW_AIR,
    enums.Action.DAMAGE_SONG,
    enums.Action.DAMAGE_SONG_RV,
    enums.Action.DAMAGE_SONG_WAIT,
    enums.Action.UNKNOWN_ANIMATION,
    enums.Action.EDGE_ATTACK_QUICK,
    enums.Action.EDGE_ATTACK_SLOW,
    enums.Action.EDGE_CATCHING,
    enums.Action.EDGE_GETUP_QUICK,
    enums.Action.EDGE_GETUP_SLOW,
    enums.Action.EDGE_HANGING,
    enums.Action.EDGE_JUMP_1_QUICK,
    enums.Action.EDGE_JUMP_1_SLOW,
    enums.Action.EDGE_JUMP_2_QUICK,
    enums.Action.EDGE_JUMP_2_SLOW,
    enums.Action.EDGE_ROLL_QUICK,
    enums.Action.EDGE_ROLL_SLOW,
    enums.Action.EDGE_TEETERING,
    enums.Action.EDGE_TEETERING_START,

      # 연구 필요
    enums.Action.GRAB_RUNNING_PULLING,
    enums.Action.PLATFORM_DROP,
    enums.Action.WALK_MIDDLE,
    enums.Action.WALK_FAST,
]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name",
    type=str,
    default="melee",
    help="name of environement"
)
parser.add_argument(
    "--continue_training",
    type=bool,
    default=False,
    help="whether to continue training with existing models",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./models/",
    help="where models are saved"
)
parser.add_argument(
    "--iso",
    type=str,
    default='../ssbm.iso',
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
args = parser.parse_args()


def run():
    """
    Start training with given options.
    """

    device = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')
    print(device)

    torch.manual_seed(500)
    np.random.seed(500)

    players = [
        PPOAgent(enums.Character.FOX, 1, 2, device, STATE_DIM, ACTION_DIM),
        NOOP(enums.Character.FOX)
        ]

    # normalizer = ObservationNormalizer(s_dim)
    if args.continue_training:
        players[0].ppo.actor_net = torch.load(
            args.model_path + "actor_net.pt").to(device)
        players[0].ppo.critic_net = torch.load(
            args.model_path + "critic_net.pt").to(device)
        # normalizer.load(args.model_path)
    else:
        with open("log_" + args.env_name + ".csv",
                  "w",
                  encoding="utf-8") as outfile:
            outfile.write("episode_id,score\n")
    episode_id = 0

    for cycle_id in range(CYCLE_NUM):
        scores = []
        players[0].ppo.buffer.buffer.clear()  # off-policy? on-policy?
        while players[0].ppo.buffer.size() < MIN_TUPLES_IN_CYCLE:
            episode_id += 1
            score = 0

            players[1] = CPU(enums.Character.FOX, random.randint(1, 9))

            env = MeleeEnv(args.iso, players, fast_forward=True)
            env.start()
            now_s, _ = env.reset(enums.Stage.FINAL_DESTINATION)

            episode_memory = []
            episode_buffer = []
            a, a_prob = None, None
            r_sum = 0
            mask_sum = 1
            action_q = []
            action_q_idx = 0
            action_pair = [0, 0]
            last_state_idx = -1
            fucked_up_cnt = 0
            for step_cnt in range(MAX_STEP):
                if step_cnt > 100:

                    if action_q_idx >= len(action_q):
                        action_q_idx = 0
                        a, a_prob = players[0].act(now_s)
                        action_q = players[0].action_space.high_action_space[a]

                    action_pair[0] = action_q[action_q_idx]
                    action_pair[1] = players[1].act(now_s[0])

                    next_s, r, done, _ = env.step(*action_pair)
                    mask = (1 - done) * 1
                    # next_state = normalizer(next_state)

                    if action_q_idx == 0:
                        episode_buffer.append([now_s, a, a_prob, step_cnt])
                    action_q_idx += 1

                    score += r
                    now_s = next_s

                    p1_action = now_s[0].players[1].action

                    if done:
                        r_sum += r
                        mask_sum *= mask
                        temp = episode_buffer[last_state_idx]
                        episode_memory.append([temp[0], temp[1], r_sum, mask_sum, temp[2]])
                        break
                    elif p1_action in players[0].action_space.sensor:
                        action_candidate = players[0].action_space.sensor[p1_action]
                        if last_state_idx < 0 or episode_buffer[last_state_idx][1] not in action_candidate:
                            for i in range(len(episode_buffer)-1, -2, -1):
                                if i <= last_state_idx:
                                    # print('There is no proper action record!', p1_action, action_candidate)
                                    # for j in range(len(episode_buffer)-1, last_state_idx, -1):
                                    #     print(episode_buffer[j][1])
                                    if last_state_idx >= 0:
                                        episode_buffer[last_state_idx][1] = action_candidate[0]
                                    fucked_up_cnt += 1
                                    break
                                if episode_buffer[i][3] > step_cnt - DELAY:
                                    continue
                                if episode_buffer[i][1] in action_candidate:
                                    if last_state_idx >= 0:
                                        temp = episode_buffer[last_state_idx]
                                        episode_memory.append([temp[0], temp[1], r_sum, mask_sum, temp[2]])
                                    r_sum = 0
                                    mask_sum = 1
                                    last_state_idx = i
                                    break
                    else:
                        if p1_action not in useless_animations:
                            print('There\'s no sensor on:', p1_action)
                    r_sum += r
                    mask_sum *= mask
                else:
                    action_pair = [0, 0]
                    now_s, _, _, _ = env.step(*action_pair)

            env.close()

            players[0].ppo.push_an_episode(episode_memory)
            print('episode:', episode_id,
                  '\tbuffer length:', players[0].ppo.buffer.size(),
                  '\tfucked up:', fucked_up_cnt)
            episode_memory = []

            with open(
                "log_" + args.env_name + ".csv", "a", encoding="utf-8"
            ) as outfile:
                outfile.write(str(episode_id) + "," + str(score) + "\n")
            scores.append(score)
        score_avg = np.mean(scores)
        print("cycle: ", cycle_id,
              "\tepisode: ", episode_id,
              "\tscore: ", score_avg)

        players[0].ppo.train()
        torch.save(
            players[0].ppo.actor_net, args.model_path + "actor_net.pt")
        torch.save(
            players[0].ppo.critic_net, args.model_path + "critic_net.pt")
        # normalizer.save(args.model_path)

    log_df = pd.read_csv("log_" + args.env_name + ".csv")
    plt.plot(log_df["episode_id"], log_df["score"])
    plt.show()


if __name__ == "__main__":
    run()
