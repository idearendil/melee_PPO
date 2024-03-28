from melee import enums
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import NOOP
import argparse
from collections import deque

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default="../ssbm.iso", type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

players = [NOOP(enums.Character.FOX), NOOP(enums.Character.FOX)]

env = MeleeEnv(args.iso, players, fast_forward=True)
env.start()

# action_buffer = deque(maxlen=3)
# action_sequence = [3, 0, 3, 0, 9, 0, 3, 0, 3, 0, 3, 0]
# action_sequence = [9]
# action_sequence = [7] * 30
# action_sequence[59] = 12
action_idx = 0

now_obs, _ = env.reset(enums.Stage.BATTLEFIELD)
for step_cnt in range(300):
    if step_cnt > 120:

        if action_idx >= len(action_sequence):
            action_pair = [0, 0]
        else:
            action_pair = [action_sequence[action_idx], 0]
            action_idx += 1

        print('control:', action_pair[0])
        next_obs, r, done, _ = env.step(*action_pair)
        print('step_cnt:', step_cnt,
              '/ action state:', next_obs[0].players[1].action)
    else:
        action_pair = [0, 0]
        next_obs, r, done, _ = env.step(*action_pair)

env.close()
