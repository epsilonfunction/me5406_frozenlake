
import numpy as np
import gym
import random

import os
from pathlib import Path
import sys
import importlib
import shutil
import time
import datetime
import argparse
import logging
import pickle

from method import actions_utils
from method.RLmethod import MonteCarlo,QLearning,SARSA



"""ARGS PARSING"""
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--method', type=str, default='ql', help='Only: sa,ql,mc; Method for Reinforcement Learning[default: ql]')
    parser.add_argument('--size', type=int, default=4,help='Size of frozen lake environment[default: 4]')
    parser.add_argument('--new_map', type=bool, default=False,help='Create new map? [default: False]')
    parser.add_argument('--map_load', type=str, default=None, help='What timestamp to load from? [default: None]')
    parser.add_argument('--episode',  type=int, default=1000,help='Total Episodes [default: 1000]')
    parser.add_argument('--alpha', type=float, default=0.9, help='Learning Rate Factor [default: 0.9]')
    parser.add_argument('--gamma', type=float, default=0.8, help='Discount Factor [default: 0.8]')
    parser.add_argument('--epsi_init',  type=float, default=1.0,help='Initial Epsilon Greedy Factor [default: 1.0]')
    parser.add_argument('--epsi_max',  type=float, default=1.0,help='Maximum Epsilon Greedy Factor [default: 1.0]')
    parser.add_argument('--epsi_min',  type=float, default=0.01,help='Minimum Epsilon Greedy Factor [default: 0.01]')
    parser.add_argument('--decay',  type=float, default=0.0001,help='Epsilon Greedy Decay Factor [default: 0.0001]')
    parser.add_argument('--default', type=bool, default=True, help='Is default map settings to be used? [default:True]')
    parser.add_argument('--render', type=bool,default=False,help="Render the learning process? [default: False]")
    return parser.parse_args()


def main(args):
    args = parse_args()

    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    timestamp = int(time.time())
    experiment_dir = Path('./data/')
    experiment_dir.mkdir(exist_ok=True)
    if args.default==True:
        experiment_dir = experiment_dir.joinpath('default/')
    else:
        if type(args.map_load) == str:
            experiment_dir = experiment_dir.joinpath(f'{args.map_load}/')
        else:
            experiment_dir = experiment_dir.joinpath(f'{timestamp}/')

    experiment_dir.mkdir(exist_ok=True)
    name_count = 0
    method_data_dump_dir = experiment_dir.joinpath(f'{args.method}%04d' % name_count)
    while method_data_dump_dir.exists() == True:
        name_count += 1
        method_data_dump_dir = experiment_dir.joinpath(f'{args.method}%04d' % name_count)
    method_data_dump_dir.mkdir()
    log_dir = method_data_dump_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.method))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    EPISODE = args.episode
    SIZE = args.size 

    if args.default==True:
        env = gym.make('FrozenLake-v1',
                       is_slippery=False,
                       render_mode=None )

    else:
        if type(args.map_load) == str:
            with open(str(experiment_dir)+'\\map_desc.pickle', 'rb') as f:
                map_desc = pickle.load(f)
        else:

            from gym.envs.toy_text.frozen_lake import generate_random_map
            map_desc=generate_random_map(size=SIZE)
            with open(str(experiment_dir)+'\\map_desc.pickle','wb') as h:
                pickle.dump(map_desc,h)

        env = gym.make('FrozenLake-v1',
                    is_slippery=False, 
                    desc=map_desc,
                    render_mode = None )
        
    action_size = env.action_space.n
    state_size = env.observation_space.n


    # rtable = np.zeros((state_size, action_size)) # Returns table
    # qtable = np.zeros((state_size, action_size))
    # ntable = np.zeros((state_size, action_size)) # For bookkeeping only

    rtable = [[[] for i in range(action_size)] for j in range(state_size)]

    NOUP = [i for i in range(SIZE)]
    NODOWN = [state_size-1-i for i in range(SIZE)]
    NOLEFT = [i*SIZE for i in range(SIZE)]
    NORIGHT = [i*SIZE + SIZE for i in range(SIZE)]

    total_episodes = args.episode # 1000
    save_interval = total_episodes//20
    alpha = args.alpha
    gamma = args.gamma

    global win, loss
    win,loss= 0,0

    # # Exponential Decay of epsilon
    epsilon = args.epsi_init
    max_epsilon = args.epsi_max
    min_epsilon = args.epsi_min
    decay_rate = args.decay

    if args.method=='ql':
        method = QLearning(SIZE,state_size,action_size,gamma,alpha)
    elif args.method=='sa':
        method = SARSA(SIZE,state_size,action_size,gamma,epsilon)
    elif args.method=='mc':
        method = MonteCarlo(SIZE,state_size,action_size,gamma)
    else:
        log_string("Invalid Method. Exiting Process.......")
        exit()

    results_array = []

    log_string(f"Method: {args.method}")
    log_string(f"Size: {SIZE} Alpha: {alpha} Gamma: {gamma}")
    log_string(f"Epsi_Init: {epsilon} Epsi_Max: {max_epsilon}")
    log_string(f"Epsi_Min: {min_epsilon} Decay: {decay_rate}")
    log_string(f"Start Time: {timestr}")

    rewards = []
    env_bounded_actions=actions_utils.boundedaction(SIZE,state_size)
    env_bounded_actions.possible_actions(4)
    for episode in range(total_episodes):

        total_rewards,reward,step = method.run_episode(env,epsilon)
        
        if reward == 1:
            win += 1
        elif reward == -1:
            loss += 1

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode+1))
        rewards.append(total_rewards)
        log_string(f"Win: {win}, Loss: {loss}")
        with open(str(method_data_dump_dir)+'\\results.txt', 'a') as g:
            g.write(f"Episode {episode+1}: Steps: {step}, Reward: {total_rewards}\n")

        if (episode+1) % save_interval == 0:
            np.save(str(log_dir)+f'\\qtable{episode}.npy',method.qtable)
            np.save(str(log_dir)+f'\\ntable{episode}.npy',method.ntable)

            MAX_STEPS = SIZE**2
        
        results_array.append([step,reward])
        print('[*] episode {}, total reward {}, average score {}'.format(episode, total_rewards, sum(rewards)/(episode+1)))

    np.save(str(method_data_dump_dir)+'\\qtable_final.npy',method.qtable)
    np.save(str(method_data_dump_dir)+'\\ntable_final.npy',method.ntable)
    ep_step_arr = np.array(results_array)
    print(ep_step_arr.shape)
    np.save(str(method_data_dump_dir)+'\\results_raw.npy',ep_step_arr)

    env.close()
if __name__ == '__main__':
    args = parse_args()
    main(args)
