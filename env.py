import typing
import random
import time

import logic
import constants as c
import gym
import numpy as np


class Game2048(gym.Env):
    def __init__(self, env_info=None):
        if env_info is None:
            env_info = {}
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, shape=(11, 4, 4), dtype='uint8')

        self.random = env_info.get("random_movements", False)
        self.human = env_info.get("human", False)
        self.matrix = logic.new_game(c.GRID_LEN)

        self.commands = {'0': logic.up, '1': logic.down,
                         '2': logic.left, '3': logic.right}

        self.score = 0
        self.rew = 0
        self.steps = 0

    def step(self, action):
        self.steps += 1
        if self.human and self.random:
            time.sleep(0.01)
        action = action if not self.random else random.randint(0, 3)
        # if self.human:
        #     print(f'Action: {["up", "down", "left", "right"][action]}')
        self.matrix, valid, self.rew = self.commands[str(action)](self.matrix)
        if valid:
            self.matrix = logic.add_two(self.matrix)
        else:
            self.rew -= 10
        state = logic.game_state(self.matrix)
        if state == 'win':
            self.rew += 10000
        elif state == 'lose':
            self.rew -= 1000
        self.score += self.rew

        info = {}
        return self.get_observation(), self.get_reward(), self.is_done(), info

    def reset(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.steps = 0
        self.score = 0
        self.rew = 0
        return self.get_observation()

    def get_observation(self):
        obs = []
        for num in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            temp2 = []
            for row in self.matrix:
                temp = []
                for cell in row:
                    temp.append(1 if cell == num else 0)
                temp2.append(temp.copy())
            obs.append(temp2.copy())
        return obs

    def get_reward(self) -> float:
        return self.rew

    def is_done(self) -> bool:
        return logic.game_state(self.matrix) in ['win', 'lose']

    def get_metrics(self) -> typing.List[float]:
        return [float(self.steps), float(self.score)]

    def render(self, mode='human'):
        obs = self.get_observation()
        print(obs)
        if not self.is_done():
            if not self.random:
                print("\nMake a move: Up: 0, Down: 1, Left: 2, Right: 3\n")
        else:
            if logic.game_state(obs) == 'win':
                print("You win!")
            else:
                print("You lose :(")
        print(f'Score: {self.score} Steps: {self.steps} Reward: {self.rew}')
