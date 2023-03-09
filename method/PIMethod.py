
import numpy as np
import gym
import random
import os

class RLmethod():
    def __init__(self,state_size,action_size) -> None:
        self.availables = []
        self.qtable = np.zeros((state_size, action_size)) # For policy improvement
        self.ntable = np.zeros((state_size, action_size)) # For bookkeeping only



    