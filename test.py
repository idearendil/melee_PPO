"""
Testing saved models within within melee environment.
"""

import argparse
import torch
import numpy as np
import psutil
from melee import enums
from parameters import MAX_STEP

# from observation_normalizer import ObservationNormalizer
from melee_env.agents.util import ObservationSpace
from melee_env.env import MeleeEnv
from melee_env.agents.basic import PPOAgent, NOOP


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, default="melee", help="name of environement"
)
parser.add_argument(
    "--model_path", type=str, default="./models/", help="where models are saved"
)
parser.add_argument(
    "--episode_num", type=int, default=100, help="How many times to test"
)
parser.add_argument(
    "--iso",
    type=str,
    default="../ssbm.iso",
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO",
)
args = parser.parse_args()


def run():
    """
    Start testing with given options.
    """

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(device)

    obs_space = ObservationSpace()

    torch.manual_seed(500)
    np.random.seed(500)

    with open("log_" + args.env_name + ".csv", "a", encoding="utf-8") as outfile:
        outfile.write("episode_id,score\n")

    players = [PPOAgent(enums.Character.FOX, device), NOOP(enums.Character.FOX)]

    # normalizer = ObservationNormalizer(s_dim)
    players[0].ppo.actor_net = torch.load(args.model_path + "actor_net.pt")
    players[0].ppo.critic_net = torch.load(args.model_path + "critic_net.pt")
    # normalizer.load(args.model_path)
    episode_id = 0

    for episode_id in range(args.episode_num):
        env = MeleeEnv(args.iso, players, fast_forward=True)
        env.start()
        gamestate, done = env.setup(enums.Stage.BATTLEFIELD)

        now_obs, _, _, _ = obs_space(gamestate, done)
        score = 0
        for _ in range(MAX_STEP):

            _, _ = players[0].act(now_obs)
            players[1].act(gamestate)
            gamestate, done = env.step()
            next_obs, r, _, _ = obs_space(gamestate, done)
            now_obs = next_obs

            score += r

            if done:
                break
        for proc in psutil.process_iter():
            if proc.name() == "Slippi Dolphin.exe":
                parent_pid = proc.pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        print("episode: ", episode_id, "\tscore: ", score)


if __name__ == "__main__":
    run()
