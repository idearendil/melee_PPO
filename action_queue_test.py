from melee import enums
from melee_env.env import MeleeEnv
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

action_buffer = deque(maxlen=3)

gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
for step_cnt in range(500):
    if step_cnt > 85:

        if step_cnt < 300:
            control = action_space(0)
            action_buffer.append(0)
            print(0)
        else:
            control = action_space(3)
            action_buffer.append(3)
            print(3)
        control(players[0].controller)

        players[1].act(gamestate)

        gamestate, done = env.step()

        next_obs, r, _, _ = observation_space(gamestate, done)
        if next_obs[3] == 20.0:
            print('action buffer:', action_buffer[0])
        print('step_cnt', step_cnt, ':', next_obs[3], next_obs[4])
    else:
        players[0].act(gamestate)
        players[1].act(gamestate)

for proc in psutil.process_iter():
    if proc.name() == "Slippi Dolphin.exe":
        parent_pid = proc.pid
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
