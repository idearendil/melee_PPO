from melee import enums
from melee_env.agents.util import ObservationSpace
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
import argparse
from arch_core import nnAgent
import torch
from torch.distributions import Categorical
import time
import os
import threading
import psutil

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default='./../ssbm.iso', type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO")
parser.add_argument("--save", default=False, type=bool)
parser.add_argument("--restore", default=False, type=bool)
parser.add_argument("--num_ep", default=10)
parser.add_argument("--kill_interval", default=2, help="env will automatically restart at kill interval")
args = parser.parse_args()

time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
MODEL_PATH = './model/'
SAVE_PATH = os.path.join(MODEL_PATH, time_str)

lr = 0.02
gamma = 0.99

def kill_process_by_name(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            pid = process.info['pid']
            try:
                # Attempt to terminate the process gracefully
                psutil.Process(pid).terminate()
                print(f"Process {process_name} (PID {pid}) terminated.")
            except psutil.NoSuchProcess:
                print(f"Process {process_name} (PID {pid}) not found.")
            except psutil.AccessDenied:
                print(f"Access denied to terminate {process_name} (PID {pid}).")
            except Exception as e:
                print(f"Error terminating {process_name} (PID {pid}): {e}")

def run():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    obs_space = ObservationSpace()
    AGENT = nnAgent(obs_space, device)
    if args.restore:
        AGENT.net.load_state_dict(torch.load(os.listdir(MODEL_PATH)[-1]))
    optimizer = torch.optim.Adam(AGENT.net.parameters(), lr=lr)
    players = [AGENT, Rest()]
    
    env = MeleeEnv(args.iso, players, fast_forward=False)
    env.start()

    for episode in range(args.num_ep):
        score = 0
        AGENT.net.core.hidden_state = None
        gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
        while not done: 
            prob = players[0].act(gamestate)
            players[1].act(gamestate)
            observation, reward, done, info = obs_space(gamestate)
            score += reward
            AGENT.rewards.append(reward)
            AGENT.probs.append(prob)
            gamestate, done = env.step()
        
        # Start thread for training
        train_ = threading.Thread(target=AGENT.train)
        train_.start()
        while train_.is_alive():
            env.console.step()
        # Wait for the thread to finish
        train_.join()

        if args.save:
            save_path = SAVE_PATH + ".pth"
            print('Save model state_dict to', save_path)
            torch.save(players[0].net.state_dict(), save_path)

        print("# of episode :{}, score : {}".format(episode + 1, score))

        if (episode + 1) % args.kill_interval == 0:
            kill_process_by_name("Slippi Dolphin.exe")
            env.start()

    kill_process_by_name("Slippi Dolphin.exe")

if __name__ == '__main__':
    run()