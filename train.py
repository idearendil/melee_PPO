"""
Start training models by PPO method within Ant-v4 environment of mujoco.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import MAX_STEP, CYCLE_NUM, MIN_STEPS_IN_CYCLE
# from observation_normalizer import ObservationNormalizer
from melee import enums
from melee_env.agents.util import ObservationSpace
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
import psutil


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name",
    type=str,
    default="melee",
    help="name of environement"
)
parser.add_argument(
    "--conitinue_training",
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

    obs_space = ObservationSpace()

    torch.manual_seed(500)
    np.random.seed(500)

    with open("log_" + args.env_name + ".csv",
              "a",
              encoding="utf-8") as outfile:
        outfile.write("episode_id,score\n")

    players = [PPOAgent(enums.Character.FOX, device), NOOP(enums.Character.FOX)]

    # normalizer = ObservationNormalizer(s_dim)
    if args.conitinue_training:
        players[0].ppo.actor_net = torch.load(
            args.model_path + "actor_net.pt").to(device)
        players[0].ppo.critic_net = torch.load(
            args.model_path + "critic_net.pt").to(device)
        # normalizer.load(args.model_path)
    episode_id = 0

    for cycle_id in range(CYCLE_NUM):
        print(cycle_id)
        scores = []
        steps_in_cycle = 0
        episode_memory = []
        players[0].ppo.buffer.buffer.clear()               # off-policy? on-policy?
        while steps_in_cycle < MIN_STEPS_IN_CYCLE:
            episode_id += 1

            env = MeleeEnv(args.iso, players, fast_forward=True)
            env.start()
            gamestate, done = env.setup(enums.Stage.BATTLEFIELD)

            now_observation, _, done, _ = obs_space(gamestate)
            score = 0
            for step_cnt in range(MAX_STEP):
                if step_cnt > 85:
                    steps_in_cycle += 1

                    a, a_prob = players[0].act(now_observation)
                    players[1].act(gamestate)

                    gamestate, done = env.step()
                    next_observation, r, done, _ = obs_space(gamestate)
                    # next_state = normalizer(next_state)

                    mask = (1 - done) * 1
                
                    episode_memory.append([now_observation, a, r, mask, a_prob])

                    score += r
                    now_observation = next_observation
                else:
                    _, _ = players[0].act(now_observation)
                    players[1].act(gamestate)
                    gamestate, done = env.step()
                    next_observation, r, done, _ = obs_space(gamestate)
                    now_observation = next_observation

                if done:
                    break

            for proc in psutil.process_iter():
                if proc.name() == "Slippi Dolphin.exe":
                    parent_pid = proc.pid
                    parent = psutil.Process(parent_pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
            players[0].ppo.push_an_episode(episode_memory)
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
        torch.save(players[0].ppo.actor_net, args.model_path + "actor_net.pt")
        torch.save(players[0].ppo.critic_net, args.model_path + "critic_net.pt")
        # normalizer.save(args.model_path)

    log_df = pd.read_csv("log_" + args.env_name + ".csv")
    plt.plot(log_df["episode_id"], log_df["score"])
    plt.show()


if __name__ == "__main__":
    run()
