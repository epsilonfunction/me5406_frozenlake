import numpy as np
from .actions_utils import epsilongreedy


class ReinforcementLearning:
    def __init__(self, size ,state_size, action_size, gamma):
        self.SIZE = size
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.qtable = np.zeros((state_size, action_size))
        self.ntable = np.zeros((state_size, action_size))
    
    def choose_action(self,epsilon,state):
        
        state_policy = epsilongreedy(self.SIZE,
                               self.state_size,
                               epsilon,
                               self.qtable[state])
        self.action = state_policy.get_action(state)
        self.ntable[state,self.action] += 1

        return self.action
    
    def update_Q(self, state, action, reward, next_state, next_action=None):
        raise NotImplementedError()
    
    def final_states(self):
        return self.qtable, self.ntable

    def run_episode(self,env):
        statestate = env.reset()
        state = statestate[0]
        env.render()

        step,total_rewards,done = 0,0,False

class MonteCarlo(ReinforcementLearning):
    def __init__(self, size, state_size, action_size, gamma):
        super().__init__(size, state_size, action_size, gamma)
        self.rtable = [[[] for i in range(action_size)] for j in range(state_size)]

    
    def update_Q(self):
        
        returns = {}
        G = 0
        for t in reversed(range(len(self.sa_seq))):
            state, action, reward = self.sa_seq[t]
            G = self.gamma * G + reward
            if str((state, action)) not in returns:
                returns[str((state, action))] = []
            else:
                continue # Avoid appending the next value if it already exists, Hence only the first value pair is added.
            returns[str((state, action))].append(G)

        for state_action, g_first in returns.items():
            state, action = eval(state_action)
            self.rtable[state][action] += g_first
            self.qtable[state, action] = np.mean(self.rtable[state][action])

    def run_episode(self,env,epsilon):

        statestate = env.reset()
        state = statestate[0]
        total_rewards = 0
        env.render()

        step,total_rewards,done = 0,0,False
        self.sa_seq = []

        action = self.choose_action(epsilon,state)
        
        while not done:
            step += 1
            new_state, reward, done, truncated, info = env.step(action) 

            self.sa_seq.append((state,action,reward))

            state = new_state
            total_rewards += reward
            self.ntable[state,action] += 1

            action = self.choose_action(epsilon,state)

        self.update_Q()
    
        return total_rewards,reward,step




class QLearning(ReinforcementLearning):
    def __init__(self, size, state_size, action_size, gamma, alpha):
        super().__init__(size ,state_size, action_size, gamma)
        self.alpha = alpha
    
    def update_Q(self, state, action, reward, new_state):
        
        self.td_Q = self.alpha*(reward+self.gamma*np.max(self.qtable[new_state]) -self.qtable[state,action])
        self.qtable[state, action] += self.td_Q

        return
    def run_episode(self,env,epsilon):
        
        statestate = env.reset()
        state = statestate[0]
        env.render()

        step,total_rewards,done = 0,0,False

        while not done:
            step += 1
            action = self.choose_action(epsilon,state)
            new_state, reward, done, truncated, info = env.step(action) 
            self.update_Q(state,
                        action,
                        reward,
                        new_state)
            
            state = new_state
            total_rewards += reward

        return total_rewards,reward,step


class SARSA(ReinforcementLearning):
    def __init__(self,  size, state_size, action_size, gamma, alpha):
        super().__init__(size ,state_size, action_size, gamma)
        self.alpha = alpha
    
    def update_Q(self, state, action, reward, new_state,new_action):
        
        self.td_S = self.alpha*(reward+self.gamma*self.qtable[new_state,new_action] -self.qtable[state,action])

        self.qtable[state, action] += self.td_S

        return
    def run_episode(self, env,epsilon):

        statestate = env.reset()
        state = statestate[0]
        env.render()
        
        action = self.choose_action(epsilon,state)

        step,total_rewards,done = 0,0,False

        while not done:
            step += 1
            new_state, reward, done, truncated, info = env.step(action) 
            new_action = self.choose_action(epsilon,new_state)
            self.update_Q(state,
                        action,
                        reward,
                        new_state,
                        new_action)
            state = new_state
            action = new_action
            total_rewards += reward

        return total_rewards,reward,step
