"""
The file of hyperparameters.
"""

LR_ACTOR = 0.0001
ENTROPY_WEIGHT = 0.0001  # 0.0001
EPSILON = 0.2

LR_CRITIC = 0.0001
L2_RATE = 0.001

GAMMA = 0.98
LAMBDA = 0.9

CYCLE_NUM = 5000
MAX_STEP = 18000
EPISODE_LEN = 100
BATCH_SIZE = 1024
MIN_TUPLES_IN_CYCLE = 100000  # 500
BUFFER_SIZE = 400000
BATCH_NUM = 150  # 60
PREDICTION_NUM = 1

PRE_STATES_NUM = 0
STATE_DIM = 811
ACTION_DIM = 45

TAU = 0.1

WIN_RATE_DECAY = 0.8

DELAY = 2
