from os import path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle

DEFAULT_MAP = np.array([
    [' ', ' ', 'R1', ' ', ' '],
    [' ', ' ', ' ', ' ', ' '],
    [' ', ' ', 'R2', ' ', ' '],
    [' ', ' ', ' ', ' ', ' '],
    ['R4', ' ', 'R3', ' ', 'H'],
])

TREASURE_MAP = {
    'R1': 80,
    'R2': 145,
    'R3': 166,
    'R4': 175,
}

class ModifiedResourceGathering(gym.Env, EzPickle):
    """
    ## Description
    From "Barrett, Leon & Narayanan, Srini. (2008). Learning all optimal policies with multiple criteria.
    Proceedings of the 25th International Conference on Machine Learning. 41-47. 10.1145/1390156.1390162."

    ## Observation Space
    The observation is discrete and consists of 4 elements:
    - 0: The x coordinate of the agent
    - 1: The y coordinate of the agent
    - 2: Flag indicating if the agent collected the gold
    - 3: Flag indicating if the agent collected the diamond

    ## Action Space
    The action is discrete and consists of 4 elements:
    - 0: Move up
    - 1: Move down
    - 2: Move left
    - 3: Move right

    ## Reward Space
    The reward is 3-dimensional:
    - 0: -1 for each step taken
    - 1: +1 if returned home with gold, else 0
    - 2: +1 if returned home with diamond, else 0

    ## Starting State
    The agent starts at the home position with no gold or diamond.

    ## Episode Termination
    The episode terminates when the agent returns home

    ## Credits
    The home asset is from https://limezu.itch.io/serenevillagerevamped
    The gold, enemy and gem assets are from https://ninjikin.itch.io/treasure
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        EzPickle.__init__(self, render_mode)

        self.render_mode = render_mode

        self.name = 'Resource Gathering'

        # The map of resource gathering env
        self.map = DEFAULT_MAP.copy()
        
        # self.initial_pos = np.array([4, 2], dtype=np.int32)
        self.initial_pos = np.array([0, 0], dtype=np.int32)
        self.final_pos   = np.array([4, 4], dtype=np.int32)
        self.current_pos = self.initial_pos.copy()

        self.dir = {
            0: np.array([-1,  0], dtype=np.int32),  # up
            1: np.array([ 1,  0], dtype=np.int32),  # down
            2: np.array([ 0, -1], dtype=np.int32),  # left
            3: np.array([ 0,  1], dtype=np.int32),  # right
        }

        self.observation_space = Box(low=0.0, high=5.0, shape=(4,), dtype=np.int32)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)

        # reward space:
        self.reward_dim = 2
        self.reward_space = Box(
            low=np.array([-2.0, 0.0]), 
            high=np.array([-100.0, 150.0]), 
            shape=(self.reward_dim,), dtype=np.float32
        )

        # pygame
        self.size = 5
        self.cell_size = (64, 64)
        self.window_size = (
            self.map.shape[1] * self.cell_size[1],
            self.map.shape[0] * self.cell_size[0],
        )
        self.clock = None
        self.elf_images = []
        self.gold_img = None
        self.gold_taken_img = None
        self.gem_img = None
        self.treasure_img = None
        self.home_img = None
        self.road_block_img = None
        self.gold_imgs = []
        self.mountain_bg_img = []
        self.window = None
        self.last_action = None

    def get_map_value(self, pos):
        return self.map[pos[0]][pos[1]]

    def is_valid_state(self, state):
        return state[0] >= 0 and state[0] < self.size and state[1] >= 0 and state[1] < self.size

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                'You are calling render method without specifying any render mode. '
                'You can specify the render_mode at initialization, '
                f'e.g. mo_gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.window is None:
            pygame.init()

            if self.render_mode == 'human':
                pygame.display.init()
                pygame.display.set_caption('Resource Gathering')
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if not self.elf_images:
                hikers = [
                    path.join(path.dirname(__file__), 'assets/char_up.png'),
                    path.join(path.dirname(__file__), 'assets/char_down.png'),
                    path.join(path.dirname(__file__), 'assets/char_left.png'),
                    path.join(path.dirname(__file__), 'assets/char_right.png'),
                ]
                self.elf_images = [pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in hikers]
            
            if not self.mountain_bg_img:
                bg_imgs = [
                    path.join(path.dirname(__file__), 'assets/bg1.png'),
                    path.join(path.dirname(__file__), 'assets/bg2.png'),
                ]

                self.mountain_bg_img = [
                    pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in bg_imgs
                ]
            
            if self.gold_taken_img is None:
                self.gold_taken_img = pygame.transform.scale(
                    pygame.image.load(path.join(path.dirname(__file__), 'assets/gold_taken.png')),
                    (0.5 * self.cell_size[0], 0.5 * self.cell_size[1]),
                )
            
            if self.treasure_img is None:
                self.treasure_img = pygame.transform.scale(
                    pygame.image.load(path.join(path.dirname(__file__), 'assets/treasure.png')),
                    (0.5 * self.cell_size[0], 0.5 * self.cell_size[1]),
                )

            if self.home_img is None:
                self.home_img = pygame.transform.scale(
                    pygame.image.load(path.join(path.dirname(__file__), 'assets/base.png')),
                    self.cell_size,
                )

            self.font = pygame.font.Font(path.join(path.dirname(__file__), 'assets', 'Minecraft.ttf'), 20)

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                check_board_mask = i % 2 ^ j % 2
                
                self.window.blit(
                    self.mountain_bg_img[check_board_mask],
                    np.array([j, i]) * self.cell_size[0],
                )
                if self.map[i, j] in ['R1', 'R2', 'R3', 'R4']:
                    self.window.blit(self.treasure_img, np.array([j + 0.25, i + 0.35]) * self.cell_size[0])

                    treasure_val = self.font.render(f'{TREASURE_MAP[self.map[i, j]]:3d}', True, (255, 255, 255))                    
                    self.window.blit(treasure_val, np.array([j + 0.25, i + 0.1]) * self.cell_size[0])
                                                
                elif self.map[i, j] == 'H':
                    self.window.blit(self.home_img, np.array([j, i]) * self.cell_size[0])
                
                elif self.map[i, j] == 'X':
                    self.window.blit(self.gold_taken_img, np.array([j + 0.22, i + 0.25]) * self.cell_size[0])
                        
        last_action = self.last_action if self.last_action is not None else 3
        self.window.blit(self.elf_images[last_action], self.current_pos[::-1] * self.cell_size[0])

        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
       
        elif self.render_mode == 'rgb_array':  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def get_state(self):
        # pos = self.current_pos.copy()
        # state = np.concatenate((
        #     pos, 
        #     np.array([self.has_gold, self.has_gem], dtype=np.int32)
        # ))
        
        return self.current_pos.copy()
    
    def set_state(self, state: np.array):
        self.current_pos = state.copy()

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.current_pos = self.initial_pos
        self.has_gem = 0
        self.has_gold = 0
        self.step_count = 0.0
        
        self.map = DEFAULT_MAP.copy()
        
        state = self.get_state()
        
        if self.render_mode == 'human':
            self.render()
        
        return state, {}

    def step(self, action):

        next_pos = self.current_pos + self.dir[int(action)]
        self.last_action = action

        if self.is_valid_state(next_pos):
            self.current_pos = next_pos

        terminal = False
        treasure_value = 0.0

        cell = self.get_map_value(self.current_pos)
        if cell in ['R1', 'R2', 'R3', 'R4']:
            
            if   cell == 'R1': treasure_value += 80
            elif cell == 'R2': treasure_value += 145
            elif cell == 'R3': treasure_value += 166
            elif cell == 'R4': treasure_value += 175

            x, y = self.current_pos
            self.map[x][y] = ' '   

        elif cell == 'H':
            treasure_value += 50
            terminal = True

        time_penalty = -1.0
        vec_reward = np.array(
            [ time_penalty, treasure_value ], 
            dtype=np.float32
        )

        state = self.get_state()
        if self.render_mode == 'human':
            self.render()
        
        return state, vec_reward, terminal, False, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    import mo_gymnasium as mo_gym

    env = mo_gym.make('modified-resource-gathering-v0', render_mode='human')
    
    terminated = False
    env.reset()

    while True:
        env.render()
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        
        if terminated or truncated:
            env.reset()
