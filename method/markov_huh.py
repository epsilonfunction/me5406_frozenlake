# Taken from : https://github.com/Urinx/ReinforcementLearning/blob/master/QLearning/QLearning_FrozenLake.py

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
experiment_dir = experiment_dir.joinpath('MC/')
experiment_dir.mkdir(exist_ok=True)
experiment_dir = experiment_dir.joinpath(str(timestamp))
experiment_dir.mkdir(exist_ok=True)

"""ENVIRONMENT"""

DEFAULT = True # True for Default settings; False for custom map and custom size
MAP_LOAD_BOOL = False

# env = gym.make('FrozenLake-v1',is_slippery=False, 
# desc=generate_random_map(size=10),render_mode="human" )
if MAP_LOAD_BOOL == False:


    SIZE = 4

    if (DEFAULT == False) or (SIZE != 4) :
        map_desc=generate_random_map(size=SIZE)
        env = gym.make('FrozenLake-v1',
                    is_slippery=False, 
                    desc=map_desc)
        with open(str(experiment_dir)+'\\map_desc.pickle','wb') as h:
            pickle.dump(map_desc,h)

    else:
        env = gym.make('FrozenLake-v1',
                    is_slippery=False)
                    # render_mode="human" )
else:
    try:
        MAP_LOAD_DIR = 'data/QL/timestamp/map_desc.pickle'
        with open(MAP_LOAD_DIR, 'rb') as f:
            map_desc = pickle.load(f)
    except:
        print("Fail to load from saved map directory. Check directory given")
        exit()

def find_duplicates(sequence):
    duplicates = {}
    for i, state_action in enumerate(sequence):
        key = str(state_action)
        if key in duplicates:
            duplicates[key].append(i)
        else:
            duplicates[key] = [i]
    return duplicates



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


action_size = env.action_space.n
state_size = env.observation_space.n
rtable = np.zeros((state_size, action_size)) # Returns table
qtable = np.zeros((state_size, action_size))
ntable = np.zeros((state_size, action_size)) # For bookkeeping only

rtable = [[[] for i in range(action_size)] for j in range(state_size)]

NOUP = [i for i in range(SIZE)]
NODOWN = [state_size-1-i for i in range(SIZE)]
NOLEFT = [i*SIZE for i in range(SIZE)]
NORIGHT = [i*SIZE + SIZE for i in range(SIZE)]


total_episodes = 10000 # 1000
save_interval = total_episodes//20
learning_rate = 0.8
max_steps = 99
gamma = 0.95

global win, loss, timeout
win,loss,timeout = 0,0,0

# # Exponential Decay of epsilon
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.005
decay_rate = 0.001

results_array = []
# Fixed P(Exploration) = epsilon Method
# epsilon = 0.01
with open(str(experiment_dir)+'\\results.txt', 'w') as f:
    f.write(f"Method: MonteCarlo \n")
    f.write(f"------------------Parameters------------------\n")
    f.write(f"Size: {SIZE} Learning Rate: {learning_rate} Gamma: {gamma}\n")
    f.write(f"Epsi_Init: {epsilon} Epsi_Max: {max_epsilon} \n")
    f.write(f"Epsi_Min: {min_epsilon} Decay: {decay_rate}\n")
    f.write(f"Start Time: {timestr} \n")

rewards = []
for episode in range(total_episodes):
    statestate = env.reset()
    state = statestate[0]
    total_rewards = 0

    sa_seq = []

    env.render()
    action_list = possible_actions(state)
    # action = env.action_space.sample()
    action = np.random.choice(action_list)

    # for step in range(max_steps):
    # For infinite runtime in MC method
    step = 0
    done = False
    while not done:
        step += 1
        new_state, reward, done, truncated, info = env.step(action) 

        # print(f"Epsiode {episode}, Step {step}")
        # print(f"state at {state}")

        sa_seq.append((state, action, reward))
        exp_exp_tradeoff = random.uniform(0, 1)

        state = new_state
        total_rewards += reward
        ntable[state,action] += 1

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
        # sa_seq.append((state,action))
        # print(f"state at {state}")
        # print(f'action at {action}')



        # print(epsilon)

        if done: 
            if reward == 1:
                win += 1
            elif reward == -1:
                loss += 1
            else:
                pass

            break
    
    # print("Episode Simulated")
    # print(sa_seq)

    # # ChatGPT
    # returns = {}
    # g = 0
    # for t in reversed(range(len(sa_seq))):
    #     state, action, reward = sa_seq[t]
    #     g = gamma * g + reward
    #     if (state, action) not in returns:
    #         returns[(state, action)] = []
    #     returns[(state, action)].append(g)

    # print(returns)
    # for state_action, g_list in returns.items():
    #     state, action = state_action
    #     print(state,action)
    #     rtable[state, action] += g_list
    #     print(rtable)
    #     qtable[state, action] = np.mean(rtable[state, action])

    # Modified
    returns = {}
    g = 0
    for t in reversed(range(len(sa_seq))):
        state, action, reward = sa_seq[t]
        g = gamma * g + reward
        if str((state, action)) not in returns:
            returns[str((state, action))] = []
        else:
            continue
        returns[str((state, action))].append(g)

    for state_action, g_list in returns.items():
        state, action = eval(state_action)
        rtable[state][action] += g_list
        qtable[state, action] = np.mean(rtable[state][action])



    # print("Qtable update done")
    # tl = len(sa_seq)
    # # dupes = find_duplicates(sa_seq)
    # # print(dupes)
    # # print(sa_seq)
    # big_G = 0
    # R = total_rewards

    # s_a = sa_seq[-1]
    # s,a = [sa_seq[-1][0]],[sa_seq[-1][1]]


    # g_seq = {}
    # g_seq[str((s,a))] = [big_G]

    # for j in range(tl-1):
    #     big_G = gamma*big_G + R
    #     idx = (-1)*(j+2)
    #     s_a = sa_seq[idx]
    #     s,a = [sa_seq[idx][0]],[sa_seq[idx][1]]
    #     # print(s,a)
    #     R= qtable[s,a]
    #     if str((s,a)) not in g_seq.keys():
    #         g_seq[str((s,a))] = [big_G]
    #     else:
    #         g_seq[str((s,a))].insert(0,big_G)

    #     # g_seq.insert(0,big_G)

    # # print(g_seq)
    # for k in g_seq.keys():

    #     kt = eval(k)
    #     # print(kt)
    #     testset = (int(kt[0][0]), int(kt[1][0]))

    #     # print(testset)
    #     firstinstval = float(g_seq[k][0])
    #     rtable[testset[0]][testset[1]].append(firstinstval)

    #     toupdateQ = np.average(
    #         rtable[testset[0]][testset[1]]        
    #     )
    #     qtable[testset[0]][testset[1]]= toupdateQ



    # print(rtable)
    
    # next_ep = input("Next Episode? : ")
    # while next_ep == False:
    #     next_ep = input("Next Episode? : ")


    # if episode % 50 == 0:
    #     np.save('qtable.npy',qtable)
    #     # np.save('ntable.npy',ntable)

    #     print(epsilon)
    #     print(qtable)
    #     # print(ntable)


    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode+1))
    print(f"Win: {win}, Loss: {loss}, Timeout:{timeout}")
    with open(str(experiment_dir)+'\\results.txt', 'a') as g:
        g.write(f"Episode {episode+1}: Steps: {step}, Reward: {total_rewards} Epsilon: {epsilon}\n")

    if (episode+1) % save_interval == 0:
        np.save(str(experiment_dir)+f'\\qtable{episode}.npy',qtable)
        np.save(str(experiment_dir)+f'\\ntable{episode}.npy',ntable)

        # print(epsilon)
        # print(qtable)
        # print(ntable)

    results_array.append([step,total_rewards])
    rewards.append(total_rewards)
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
# # Final Demo, Showcase Best Scenario
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