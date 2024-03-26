"""
Start training models by PPO method within melee environment.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from melee import enums
from parameters import MAX_STEP, CYCLE_NUM, MIN_STEPS_IN_CYCLE
# from observation_normalizer import ObservationNormalizer
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP


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

    players = [PPOAgent(enums.Character.FOX, device),
               NOOP(enums.Character.FOX)]

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
        steps_in_cycle = 0
        episode_memory = []
        players[0].ppo.buffer.buffer.clear()  # off-policy? on-policy?
        while steps_in_cycle < MIN_STEPS_IN_CYCLE:
            episode_id += 1
            score = 0

            env = MeleeEnv(args.iso, players, fast_forward=True)
            env.start()
            now_s, _ = env.reset(enums.Stage.BATTLEFIELD)
            for step_cnt in range(MAX_STEP):
                if step_cnt > 100:
                    steps_in_cycle += 1

                    action_pair = [0, 0]
                    a, a_prob = players[0].act(now_s)
                    action_pair[0] = a
                    action_pair[1] = players[1].act()

                    next_s, r, done, _ = env.step(*action_pair)
                    # next_state = normalizer(next_state)

                    mask = (1 - done) * 1

                    episode_memory.append([now_s, a, r, mask, a_prob])

                    score += r
                    now_s = next_s

                    if done:
                        break
                else:
                    action_pair = [0, 0]
                    now_obs, _, _, _ = env.step(*action_pair)

            env.close()

            players[0].ppo.push_an_episode(episode_memory)
            print('episode:', episode_id,
                  'buffer length:', players[0].ppo.buffer.size())
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
