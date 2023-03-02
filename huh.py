# Taken from : https://github.com/Urinx/ReinforcementLearning/blob/master/QLearning/QLearning_FrozenLake.py

import numpy as np
import gym
import random

from gym.envs.toy_text.frozen_lake import generate_random_map

# env = gym.make('FrozenLake-v1', desc=generate_random_map(size=10),render_mode="human" )
env = gym.make('FrozenLake-v1',is_slippery=False,render_mode="human" )

action_size = env.action_space.n
state_size = env.observation_space.n
reward_sp = env.reward_range
print(reward_sp)
qtable = np.zeros((state_size, action_size))
ntable = np.zeros((state_size, action_size)) # For bookkeeping only

total_episodes = 200 # 1000
learning_rate = 0.8
max_steps = 99
gamma = 0.95

global win, loss, timeout
win,loss,timeout = 0,0,0

# Exponential Decay of epsilon
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# Fixed P(Exploration) = epsilon Method
# epsilon = 0.01

rewards = []
for episode in range(total_episodes):
    statestate = env.reset()
    state = statestate[0]
    total_rewards = 0

    env.render()

    for step in range(max_steps):
    # For infinite runtime in MC method
    # step = 0
    # while True:
        # step += 1
        print(f"Epsiode {episode}, Step {step}")
        print(f"state at {state}")


        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state])
        else:
            action = env.action_space.sample()

        # action = input("Action Here: ")

        # for i in range(4):
        #     print(env.action_space.sample())
        # env.render()

        # action = int(action)
        new_state, reward, done, truncated, info = env.step(action) 
        
        # input()
        print(f"new_state, reward, done, truncated, info are: {new_state, reward, done, truncated, info}")
        print(f'Old Q Table: {qtable}')
        print(f'Old N Table: {ntable}')

        print(f"state at {state}")
        print(f'action at {action}')

        ntable[state,action] += 1
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        # print(qtable)
        # print(ntable)

        # input()
        print(f'New Q Table: {qtable}')
        print(f'New N Table: {ntable}')

        state = new_state
        total_rewards += reward

        # print(epsilon)
        if step == max_steps-1:
            timeout += 1

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
    print(f"Win: {win}, Loss: {loss}, Timeout:{timeout}")

    if episode % 50 == 0:
        np.save('qtable.npy',qtable)
        np.save('ntable.npy',ntable)

        print(epsilon)
        print(qtable)
        print(ntable)

    print('[*] episode {}, total reward {}, average score {}'.format(episode, total_rewards, sum(rewards)/(episode+1)))

print(qtable)
print(ntable)

# np.save('qtable_final.npy',qtable)
# np.save('ntable_final.npy',ntable)

# Play the game

# for episode in range(1):
#     statestate = env.reset()
#     state = statestate[0]
#     print('*'*20)
#     print('EPISODE ', episode)

#     for step in range(max_steps):
#         env.render()
#         action = np.argmax(qtable[state])
#         input()
#         state, reward, done, info = env.step(action)
#         if done: break

env.close()