from melee import enums
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import NOOP
import argparse
import psutil
from melee_env.agents.util import ActionSpace, ObservationSpace
from collections import deque

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default="../ssbm.iso", type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

players = [NOOP(enums.Character.FOX), NOOP(enums.Character.FOX)]

env = MeleeEnv(args.iso, players, fast_forward=True)
env.start()

action_space = ActionSpace()
observation_space = ObservationSpace()

# action_buffer = deque(maxlen=3)
action_sequence = 

now_obs, _ = env.reset(enums.Stage.BATTLEFIELD)
for step_cnt in range(500):
    if step_cnt > 85:

        action_pair = [0, 0]

        next_obs, r, done, _ = env.step(*action_pair)
        if next_obs[3] == 20.0:
            print('action buffer:', action_buffer[0])
        print('step_cnt', step_cnt, ':', next_obs[3], next_obs[4])

env.close()
