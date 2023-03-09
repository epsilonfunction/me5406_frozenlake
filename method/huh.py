# Taken from : https://github.com/Urinx/ReinforcementLearning/blob/master/QLearning/QLearning_FrozenLake.py
# QLearning

import numpy as np
import gym
import random
from pathlib import Path
import datetime
import time
import pickle

from gym.envs.toy_text.frozen_lake import generate_random_map

"""LOGGING"""
timestamp = int(time.time())
timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
experiment_dir = Path('./data/')
experiment_dir.mkdir(exist_ok=True)
experiment_dir = experiment_dir.joinpath('QL/')
experiment_dir.mkdir(exist_ok=True)
experiment_dir = experiment_dir.joinpath(str(timestamp))
experiment_dir.mkdir(exist_ok=True)

"""ENVIRONMENT"""

NEW_MAP = True # Use Default settings; False for custom map and custom size
MAP_LOAD_BOOL = False

# env = gym.make('FrozenLake-v1',is_slippery=False, 
# desc=generate_random_map(size=10),render_mode="human" )
if MAP_LOAD_BOOL == False:


    SIZE = 4

    if (NEW_MAP == False) or (SIZE != 4) :
        map_desc=generate_random_map(size=SIZE)
        env = gym.make('FrozenLake-v1',
                    is_slippery=False, 
                    desc=map_desc)
        with open(str(experiment_dir)+'\\map_desc.pickle','wb') as h:
            pickle.dump(map_desc,h)

    else:
        env = gym.make('FrozenLake-v1',
                    is_slippery=False,
                    render_mode=None )
else:
    try:
        MAP_LOAD_DIR = 'data/QL/timestamp/map_desc.pickle'
        with open(MAP_LOAD_DIR, 'rb') as f:
            my_list = pickle.load(f)
    except:
        print("Fail to load from saved map directory. Check directory given")
        exit()



action_size = env.action_space.n
state_size = env.observation_space.n
reward_sp = env.reward_range
print(reward_sp)
qtable = np.zeros((state_size, action_size))
ntable = np.zeros((state_size, action_size)) # For bookkeeping only


NOUP = [i for i in range(SIZE)]
NODOWN = [state_size-1-i for i in range(SIZE)]
NOLEFT = [i*SIZE for i in range(SIZE)]
NORIGHT = [i*SIZE + SIZE for i in range(SIZE)]

def possible_actions(state):
    action_list = [0,1,2,3]
    if state in NOLEFT:
        action_list.remove(0)
    if state in NODOWN:
        action_list.remove(1)
    if state in NORIGHT:
        action_list.remove(2)
    if state in NOUP:
        action_list.remove(3)
    return action_list

total_episodes = 10000 # 1000
save_interval = total_episodes//20
learning_rate = 0.8
max_steps = 99
gamma = 0.95

global win, loss, timeout
win,loss,timeout = 0,0,0

# Exponential Decay of epsilon
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.001

results_array = []
with open(str(experiment_dir)+'\\results.txt', 'w') as f:
    f.write(f"Method: Q Learning \n")
    f.write(f"Start Time: {timestr} \n")


# Fixed P(Exploration) = epsilon Method
# epsilon = 0.1
# for state in range(state_size):
#     if state in NOLEFT:
#         qtable[state,0] = -2
#     if state in NODOWN:
#         qtable[state,1] = -2
#     if state in NORIGHT:
#         qtable[state,2] = -2
#     if state in NOUP:
#         qtable[state,3] = -2

rewards = []
for episode in range(total_episodes):
    statestate = env.reset()
    state = statestate[0]
    total_rewards = 0

    env.render()

    # for step in range(max_steps):
    # For infinite runtime in MC method
    step = 0
    while True:
        step += 1
        # print(f"Epsiode {episode}, Step {step}")
        # print(f"state at {state}")


        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            # action = np.argmax(qtable[state])
            try:
                action_list = possible_actions(state) 
                row = list(qtable[state])
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
                action = np.argmax(qtable[state])

        else:
            action_list = possible_actions(state)
            # action = env.action_space.sample()
            action = np.random.choice(action_list)
        # action = input("Action Here: ")

        # for i in range(4):
        #     print(env.action_space.sample())
        # env.render()

        # action = int(action)
        new_state, reward, done, truncated, info = env.step(action) 
        
        # input()
        # print(f"new_state, reward, done, truncated, info are: {new_state, reward, done, truncated, info}")
        # print(f'Old Q Table: {qtable}')
        # print(f'Old N Table: {ntable}')

        # print(f"state at {state}")
        # print(f'action at {action}')

        ntable[state,action] += 1
        td_Q = learning_rate * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action]) #Temporal Difference: Q Learning
        # td_S = learning_rate * (reward + gamma * qtable[new_state][action_t+1] - qtable[state, action])


        qtable[state, action] += td_Q
        # print(qtable)
        # print(ntable)

        # input()
        # print(f'New Q Table: {qtable}')
        # print(f'New N Table: {ntable}')

        state = new_state
        total_rewards += reward

        # print(epsilon)

        if done: 
            if reward == 1:
                win += 1
            elif reward == -1:
                loss += 1
            else:
                pass

            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode+1))
    rewards.append(total_rewards)
    print(f"Win: {win}, Loss: {loss}")
    with open(str(experiment_dir)+'\\results.txt', 'a') as g:
        g.write(f"Episode {episode+1}: Steps: {step}, Reward: {total_rewards}\n")

    if (episode+1) % save_interval == 0:
        np.save(str(experiment_dir)+f'\\qtable{episode}.npy',qtable)
        np.save(str(experiment_dir)+f'\\ntable{episode}.npy',ntable)

        # print(epsilon)
        # print(qtable)
        # print(ntable)
    results_array.append([step,reward])
    print('[*] episode {}, total reward {}, average score {}'.format(episode, total_rewards, sum(rewards)/(episode+1)))

# print(qtable)
# print(ntable)
np.save(str(experiment_dir)+'\\qtable_final.npy',qtable)
np.save(str(experiment_dir)+'\\ntable_final.npy',ntable)
ep_step_arr = np.array(results_array)
print(ep_step_arr.shape)
np.save(str(experiment_dir)+'\\results_raw.npy',ep_step_arr)

# np.save('qtable_final.npy',qtable)
# np.save('ntable_final.npy',ntable)

# Play the game

# tocontinue = input("Proceed with Final Demonstration?")

# for episode in range(1):
#     statestate = env.reset()
#     state = statestate[0]
#     print('*'*20)
#     print('EPISODE ', episode)

#     for step in range(max_steps):
#         env.render()
#         action = np.argmax(qtable[state])
#         new_state, reward, done, truncated, info = env.step(action) 
        
#         if done: break

env.close()