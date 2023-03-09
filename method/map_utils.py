import numpy as np
import gym
import random
import pickle
import os
from pathlib import Path

from gym.envs.toy_text.frozen_lake import is_valid

def fix_map(lst,p_hole=0.25):
    hashmap={}
    size_count = 0
    length = len(lst)

    print(lst)

    for i in range(length): #row
        for j in range(length): #column
            size_count+= 1
            letter =lst[i][j]
            if letter not in hashmap.keys():
                hashmap[letter] = [(i,j)]
            else:
                hashmap[letter].append((i,j))
    oldmap = lst.copy()
    ideal = p_hole*size_count  
    if len(hashmap["H"]) < ideal:
        deficit = ideal - len(hashmap["H"])
        print(f"Deficit: {deficit}")
        for i in range(int(deficit)):
            rpl_idx = np.random.choice(len(hashmap["F"]))
            to_replace = hashmap["F"][rpl_idx]
            newmap = oldmap.copy()
            row = newmap[0]
            try:
                row = row[:to_replace[1]] + "H" + row[to_replace[1]+1:]
            except:
                pass
            newmap[to_replace[0]] = row
            print(newmap)
            newmapbool = is_valid(new_map,len(lst))
            while newmapbool == False:
                input(f"{i}")
                to_replace = np.random.choice(hashmap["F"])
                newmap = oldmap.copy()
                row = newmap[0]
                try:
                    row = row[:to_replace[1]] + "H" + row[to_replace[1]+1:]
                except:
                    pass
                newmap[to_replace[0]] = row
                print(newmap)
                newmapbool = is_valid(new_map,len(lst))
            
            oldmap=newmap.copy()
            hashmap["H"].append(to_replace)
            hashmap["F"].remove(to_replace)
    elif len(hashmap["H"]) > ideal:
        deficit = len(hashmap["H"]) - ideal
        print(f"Deficit: {deficit}")
        for i in range(int(deficit)):
            rpl_idx = np.random.choice(len(hashmap["H"]))
            to_replace = hashmap["H"][rpl_idx]
            newmap = oldmap.copy()
            row = newmap[0]
            try:
                row = row[:to_replace[1]] + "F" + row[to_replace[1]+1:]
            except:
                pass
            newmap[to_replace[0]] = row
            print(newmap)
            newmapbool = is_valid(new_map,len(lst))
            while newmapbool == False:
                input(f"{i}")
                rpl_idx = np.random.choice(len(hashmap["H"]))
                to_replace = hashmap["H"][rpl_idx]
                newmap = oldmap.copy()
                row = newmap[0]
                try:
                    row = row[:to_replace[1]] + "F" + row[to_replace[1]+1:]
                except:
                    pass
                newmap[to_replace[0]] = row
                print(newmap)
                newmapbool = is_valid(new_map,len(lst))

            oldmap=newmap.copy()
            hashmap["F"].append(to_replace)
            hashmap["H"].remove(to_replace)

    else:
        return lst
    lst = newmap
    return lst




import numpy as np

def select_floors_to_replace(lake):
    """
    Randomly selects 3 floors to replace with holes, while ensuring the path remains valid.
    Avoids replacing the start or end floors.
    """
    n, m = len(lake),len(lake[0])
    # Select 3 random floors to replace
    replace_indices = np.random.choice(range(1, n-1), size=3, replace=False)
    # Check if any of the selected floors are on the path
    path_indices = np.where(lake == 0)[0]
    if any(idx in path_indices for idx in replace_indices):
        # If any of the selected floors are on the path, select new ones
        return select_floors_to_replace(lake)
    else:
        return replace_indices

def custom_map(lst):
    width,height = len(lst),len(lst[0])

    assert width != height,"Map provided must be square"

    # frozen_lake is_valid requires the start to be from 0,0
    # Custom Start is not allowed
    # Custom end is allowed

    assert is_valid(lst) == True, "Map Provided does not meet frozen_lake requirements"

    mapname = str(input("Input Map Name Here"))
    experiment_dir=f"../data/{mapname}"
    with open(str(experiment_dir)+f'/{mapname}.pickle','wb') as h:
        pickle.dump(lst,h)


def generate_frozen_lake_map(size, hole_prob):
    """
    Generates a random FrozenLake environment map with the specified size and hole probability.
    
    Parameters:
    - size: integer, size of the square environment map
    - hole_prob: float, probability of a tile being a hole, including the start and end tiles
    
    Returns:
    - map_list: list of strings, FrozenLake environment map
    """
    # Define start and goal positions
    start_pos = (0, 0)
    goal_pos = (size - 1, size - 1)

    # Generate a list of tiles with the correct number of holes
    num_holes = int((size * size - 2) * hole_prob)
    tiles = ['S'] + ['F'] * (size * size - 2 - num_holes) + ['H'] * num_holes + ['G']
    random.shuffle(tiles)
    
    # Convert the list of tiles into a map
    map_list = [tiles[i:i+size] for i in range(0, len(tiles), size)]
    
    # Set the start and end positions
    map_list[start_pos[0]][start_pos[1]] = 'S'
    map_list[goal_pos[0]][goal_pos[1]] = 'G'
    
    return map_list

def is_valid_frozen_lake_map(map_list):
    """
    Checks if the specified FrozenLake environment map is valid.
    
    Parameters:
    - map_list: list of strings, FrozenLake environment map
    
    Returns:
    - is_valid: boolean, True if map is valid, False otherwise
    """
    size = len(map_list)
    
    # Check if start and goal tiles are valid
    if map_list[0][0] != 'S' or map_list[size-1][size-1] != 'G':
        return False
    
    # Check if all tiles are either 'S', 'G', 'F', or 'H'
    tiles = set(''.join(str(row) for row in map_list))
    if tiles - set('SGFH'):
        return False
    
    # Check if there is a path from start to goal tile
    visited = set()
    visited.add((0, 0))
    stack = [(0, 0)]
    
    while len(stack) > 0:
        row, col = stack.pop()
        
        if row == size-1 and col == size-1:
            return True
        
        neighbors = []
        if row > 0: neighbors.append((row-1, col))
        if row < size-1: neighbors.append((row+1, col))
        if col > 0: neighbors.append((row, col-1))
        if col < size-1: neighbors.append((row, col+1))
        
        for neighbor in neighbors:
            n_row, n_col = neighbor
            if (n_row, n_col) not in visited and map_list[n_row][n_col] in ['S', 'G', 'F']:
                visited.add((n_row, n_col))
                stack.append(neighbor)
    
    return False

if __name__=="__main__":
    new_map=["SFHF",
             "HFFF",
             "HFHF",
             "HFFG"]
    absolutely_new_map = fix_map(new_map)
    print(absolutely_new_map)