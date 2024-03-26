import numpy as np
import melee
from collections import deque


class ObservationSpace:
    def __init__(self):
        self.previous_gamestate = None
        self.current_gamestate = None
        self.curr_action = None
        self.player_count = None
        self.previous_actions = deque(maxlen=3)
        self.previous_actions.extend((0, 0, 0))

    def set_player_keys(self, keys):
        self.player_keys = keys

    def get_stocks(self, gamestate):
        stocks = [gamestate.players[i].stock for i in list(gamestate.players.keys())]
        return np.array([stocks], dtype=np.float32)

    def get_actions(self, gamestate):
        actions = np.zeros((len(gamestate.players.keys()), 386), dtype=np.float32)
        if gamestate.players[1].action.value < 386:
            actions[0, gamestate.players[1].action.value] = 1.0
        if gamestate.players[2].action.value < 386:
            actions[1, gamestate.players[2].action.value] = 1.0
        return actions.reshape((-1,))

    def get_action_frames(self, gamestate):
        action_frames = [
            gamestate.players[i].action_frame for i in list(gamestate.players.keys())
        ]
        return np.array(action_frames, dtype=np.float32)

    def get_speeds(self, gamestate):
        speed_air_x_self = [
            gamestate.players[i].speed_air_x_self
            for i in list(gamestate.players.keys())
        ]
        speed_ground_x_self = [
            gamestate.players[i].speed_ground_x_self
            for i in list(gamestate.players.keys())
        ]
        speed_x_attack = [
            gamestate.players[i].speed_x_attack for i in list(gamestate.players.keys())
        ]
        speed_y_attack = [
            gamestate.players[i].speed_y_attack for i in list(gamestate.players.keys())
        ]
        speed_y_self = [
            gamestate.players[i].speed_y_self for i in list(gamestate.players.keys())
        ]
        return np.array(sum([speed_air_x_self,
                             speed_ground_x_self,
                             speed_x_attack,
                             speed_y_attack,
                             speed_y_self], []), dtype=np.float32)

    def get_states(self, gamestate):
        facing = [gamestate.players[i].facing for i in list(gamestate.players.keys())]
        hitstun_frames_left = [
            gamestate.players[i].hitstun_frames_left
            for i in list(gamestate.players.keys())
        ]
        invulnerability_frames_left = [
            gamestate.players[i].invulnerability_left
            for i in list(gamestate.players.keys())
        ]
        jumps_left = [
            gamestate.players[i].jumps_left for i in list(gamestate.players.keys())
        ]
        off_stage = [
            gamestate.players[i].off_stage for i in list(gamestate.players.keys())
        ]
        on_ground = [
            gamestate.players[i].on_ground for i in list(gamestate.players.keys())
        ]
        percent = [gamestate.players[i].percent for i in list(gamestate.players.keys())]
        shield_strength = [
            gamestate.players[i].shield_strength for i in list(gamestate.players.keys())
        ]
        return np.array(sum([facing,
                             hitstun_frames_left,
                             invulnerability_frames_left,
                             jumps_left,
                             off_stage,
                             on_ground,
                             percent,
                             shield_strength], []), dtype=np.float32)

    def get_positions(self, gamestate):
        x_positions = [
            gamestate.players[i].position.x for i in list(gamestate.players.keys())
        ]
        y_positions = [
            gamestate.players[i].position.y for i in list(gamestate.players.keys())
        ]
        return np.array(sum([x_positions,
                             y_positions], []), dtype=np.float32)

    def get_relative_distance(self, gamestate):
        if len(gamestate.players.keys()) > 2:
            return np.zeros((len(gamestate.players.keys()), 1))
        distances = [
            gamestate.players[1].position.x - gamestate.players[2].position.x,
            gamestate.players[1].position.y - gamestate.players[2].position.y,
        ]
        return np.array(distances, dtype=np.float32)

    def get_previous_actions(self, action):
        actions = np.zeros((45 * 3,), dtype=np.float32)
        self.previous_actions.append(action)
        for i in range(3):
            actions[i * 45 + self.previous_actions[i]] = 1.0
        return actions

    def __call__(self, gamestate, action):
        reward = 0
        info = None
        self.current_gamestate = gamestate
        self.player_count = len(list(gamestate.players.keys()))

        observation = np.concatenate((self.get_positions(gamestate),
                                      self.get_relative_distance(gamestate),
                                      self.get_states(gamestate),
                                      self.get_actions(gamestate),
                                      self.get_action_frames(gamestate),
                                      self.get_speeds(gamestate),
                                      self.get_previous_actions(action)), axis=0)

        # if self.previous_gamestate is not None:
        #     p1_dmg = (
        #         self.current_gamestate.players[1].percent
        #         - self.previous_gamestate.players[1].percent
        #     )
        #     p1_shield_dmg = (
        #         self.previous_gamestate.players[1].shield_strength
        #         - self.current_gamestate.players[1].shield_strength
        #     )
        #     p1_stock_loss = int(self.previous_gamestate.players[1].stock) - int(
        #         self.current_gamestate.players[1].stock
        #     )
        #     p2_dmg = (
        #         self.current_gamestate.players[2].percent
        #         - self.previous_gamestate.players[2].percent
        #     )
        #     p2_shield_dmg = (
        #         self.previous_gamestate.players[2].shield_strength
        #         - self.current_gamestate.players[2].shield_strength
        #     )
        #     p2_stock_loss = int(self.previous_gamestate.players[2].stock) - int(
        #         self.current_gamestate.players[2].stock
        #     )

        #     p1_dmg = max(p1_dmg, 0)
        #     p2_dmg = max(p2_dmg, 0)
        #     if p1_stock_loss > 1:
        #         p1_stock_loss = 0
        #     if p2_stock_loss > 1:
        #         p2_stock_loss = 0
        #     p1_stock_loss = max(p1_stock_loss, 0)
        #     p2_stock_loss = max(p2_stock_loss, 0)

        #     w_dmg, w_shield, w_stock = 0.1, 0.02, 10
        #     p1_loss = (
        #         w_dmg * p1_dmg + w_shield * p1_shield_dmg + w_stock * p1_stock_loss
        #     )
        #     p2_loss = (
        #         w_dmg * p2_dmg + w_shield * p2_shield_dmg + w_stock * p2_stock_loss
        #     )

        #     reward = p2_loss - p1_loss
        # else:
        #     reward = 0

        if self.previous_gamestate is not None:
            p1_stock_loss = int(self.previous_gamestate.players[1].stock) - int(
                self.current_gamestate.players[1].stock
            )
            reward = p1_stock_loss
        else:
            reward = 0

        self.previous_gamestate = self.current_gamestate

        stocks = np.array([gamestate.players[i].stock for i in list(
            gamestate.players.keys())])
        done = not np.sum(stocks[np.argsort(stocks)][::-1][1:])

        return observation, reward, done, info

    def reset(self):
        self.__init__()
        # print("observation space got reset!")


class ActionSpace:
    def __init__(self):
        mid = np.sqrt(2) / 2

        self.stick_space_reduced = np.array(
            [
                [0.0, 0.0],  # no op
                [0.0, 1.0],
                [mid, mid],
                [1.0, 0.0],
                [mid, -mid],
                [0.0, -1.0],
                [-mid, -mid],
                [-1.0, 0.0],
                [-mid, mid],
            ]
        )

        self.button_space_reduced = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Action space size is total number of possible actions. In this case,
        #    is is all possible main stick positions * all c-stick positions *
        #    all the buttons. A normal controller has ~51040 possible main stick
        #    positions. Each trigger has 255 positions. The c-stick can be
        #    reduced to ~5 positions. Finally, if all buttons can be pressed
        #    in any combination, that results in 32 combinations. Not including
        #    the dpad or start button, that is 51040 * 5 * 255 * 2 * 32 which
        #    is a staggering 4.165 billion possible control states.

        # Given this, it is reasonable to reduce this. In the above class, the
        #    main stick has been reduced to the 8 cardinal positions plus the
        #    center (no-op). Only A, B, Z, and R are used, as these correspond
        #    to major in-game functions (attack, special, grab, shield). Every
        #    action can theoretically be performed with just these buttons. A
        #    final "button" is added for no-op.
        #
        #    Action space = 9 * 5 = 45 possible actions.
        self.action_space = np.zeros(
            (self.stick_space_reduced.shape[0] * self.button_space_reduced.shape[0], 3)
        )

        for button in self.button_space_reduced:
            self.action_space[int(button) * 9 : (int(button) + 1) * 9, :2] = (
                self.stick_space_reduced
            )
            self.action_space[int(button) * 9 : (int(button) + 1) * 9, 2] = button

        # self.action_space will look like this, where the first two columns
        #   represent the control stick's position, and the final column is the
        #   currently depressed button.

        # ACT  Left/Right    Up/Down      Button
        # ---  ------        ------       ------
        # 0   [ 0.        ,  0.        ,  0. (NO-OP)] Center  ---
        # 1   [ 0.        ,  1.        ,  0.        ] Up         |
        # 2   [ 0.70710678,  0.70710678,  0.        ] Up/Right   |
        # 3   [ 1.        ,  0.        ,  0.        ] Right      |
        # 4   [ 0.70710678, -0.70710678,  0.        ] Down/Right |- these repeat
        # 5   [ 0.        , -1.        ,  0.        ] Down       |
        # 6   [-0.70710678, -0.70710678,  0.        ] Down/Left  |
        # 7   [-1.        ,  0.        ,  0.        ] Left       |
        # 8   [-0.70710678,  0.70710678,  0.        ] Up/Left  ---
        # 9   [ 0.        ,  0.        ,  1. (A)    ]
        # 10  [ 0.        ,  1.        ,  1.        ]
        # 11  [ 0.70710678,  0.70710678,  1.        ]
        # 12  [ 1.        ,  0.        ,  1.        ]
        # 13  [ 0.70710678, -0.70710678,  1.        ]
        # 14  [ 0.        , -1.        ,  1.        ]
        # 15  [-0.70710678, -0.70710678,  1.        ]
        # 16  [-1.        ,  0.        ,  1.        ]
        # 17  [-0.70710678,  0.70710678,  1.        ]
        # 18  [ 0.        ,  0.        ,  2. (B)    ]
        # 19  [ 0.        ,  1.        ,  2.        ]
        # 20  [ 0.70710678,  0.70710678,  2.        ]
        # 21  [ 1.        ,  0.        ,  2.        ]
        # 22  [ 0.70710678, -0.70710678,  2.        ]
        # 23  [ 0.        , -1.        ,  2.        ]
        # 24  [-0.70710678, -0.70710678,  2.        ]
        # 25  [-1.        ,  0.        ,  2.        ]
        # 26  [-0.70710678,  0.70710678,  2.        ]
        # 27  [ 0.        ,  0.        ,  3. (Z)    ]
        # 28  [ 0.        ,  1.        ,  3.        ]
        # 29  [ 0.70710678,  0.70710678,  3.        ]
        # 30  [ 1.        ,  0.        ,  3.        ]
        # 31  [ 0.70710678, -0.70710678,  3.        ]
        # 32  [ 0.        , -1.        ,  3.        ]
        # 33  [-0.70710678, -0.70710678,  3.        ]
        # 34  [-1.        ,  0.        ,  3.        ]
        # 35  [-0.70710678,  0.70710678,  3.        ]
        # 36  [ 0.        ,  0.        ,  4. (R)    ]
        # 37  [ 0.        ,  1.        ,  4.        ]
        # 38  [ 0.70710678,  0.70710678,  4.        ]
        # 39  [ 1.        ,  0.        ,  4.        ]
        # 40  [ 0.70710678, -0.70710678,  4.        ]
        # 41  [ 0.        , -1.        ,  4.        ]
        # 42  [-0.70710678, -0.70710678,  4.        ]
        # 43  [-1.        ,  0.        ,  4.        ]
        # 45  [-0.70710678,  0.70710678,  4.        ]

        self.size = self.action_space.shape[0]

    def sample(self):
        return np.random.choice(self.size)

    def __call__(self, action):
        if action > self.size - 1:
            exit("Error: invalid action!")

        return ControlState(self.action_space[action])


class ControlState:
    def __init__(self, state):
        self.state = state
        self.buttons = [
            False,
            melee.enums.Button.BUTTON_A,
            melee.enums.Button.BUTTON_B,
            melee.enums.Button.BUTTON_Z,
            melee.enums.Button.BUTTON_R,
        ]

    def __call__(self, controller):
        controller.release_all()
        if self.state[2]:  # only press button if not no-op
            if self.state[2] != 4.0:  # special case for r shoulder
                controller.press_button(self.buttons[int(self.state[2])])
            else:
                controller.press_shoulder(melee.enums.Button.BUTTON_R, 1)

        controller.tilt_analog_unit(
            melee.enums.Button.BUTTON_MAIN, self.state[0], self.state[1]
        )


def from_observation_space(act):
    def get_observation(self, *args):
        gamestate = args[0]
        observation = self.observation_space(gamestate)
        return act(self, observation)

    return get_observation


def from_action_space(act):
    def get_action_encoding(self, *args):
        gamestate = args[0]
        action = act(self, gamestate)
        control = self.action_space(action)
        control(self.controller)
        return

    return get_action_encoding
