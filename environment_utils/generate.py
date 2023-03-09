
import numpy as np
import gym
import random

from gym.envs.toy_text.frozen_lake import generate_random_map

class environment():
    def __init__(self,size=4) -> None:
        
        self.env = gym.make('FrozenLake-v1', 
                       desc=generate_random_map(size=size),
                       render_mode="human" 
                       )
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n

    
    def train(self,method='markov',save_directory=None):
        pass

    def 
