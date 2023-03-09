import numpy as np
import random as random

class boundedaction():
    def __init__(self,SIZE,state_size) -> None:
        self.NOUP = [i for i in range(SIZE)]
        self.NODOWN = [state_size-1-i for i in range(SIZE)]
        self.NOLEFT = [i*SIZE for i in range(SIZE)]
        self.NORIGHT = [i*SIZE + SIZE for i in range(SIZE)]
    
    def possible_actions(self,state):
        action_list = [0,1,2,3]
        if state in self.NOLEFT:
            action_list.remove(0)
        if state in self.NODOWN:
            action_list.remove(1)
        if state in self.NORIGHT:
            action_list.remove(2)
        if state in self.NOUP:
            action_list.remove(3)
        return action_list
    
    def get_action(self,state):
        raise NotImplementedError()

class epsilongreedy(boundedaction):
    def __init__(self, SIZE, state_size,epsilon,qtable_state) -> None:
        super().__init__(SIZE, state_size)
        self.epsilon = epsilon
        self.qtable_state = qtable_state

    def get_action(self,state):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > self.epsilon:
            # action = np.argmax(qtable[state])
            try:
                action_list = self.possible_actions(state) 
                row = list(self.qtable_state)
                goodq = []
                for i in action_list:
                    goodq.append(row[i])

                max_value = np.max(goodq)
                max_indices = np.where(row == max_value)[0]
                if len(max_indices) == 1:
                    action = max_indices[0]
                else:
                    # if there are multiple maximum indices, choose one randomly
                    action = np.random.choice(max_indices)
            except:
                action = np.argmax(self.qtable_state[state])

        else:
            action_list = self.possible_actions(state)
            # action = env.action_space.sample()
            action = np.random.choice(action_list)

        return action #Returns from [0,1,2,3]