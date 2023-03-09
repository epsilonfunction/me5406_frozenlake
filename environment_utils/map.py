import numpy as np
import matplotlib.pyplot as plt

class cell:
    def __init__(self,location,reward = 0) -> None:
        self.name = 'cell'
        self.location = location
        self.reward = reward
    # def __getattribute__(self, __name: str) -> Any:
    #     pass

class state(cell):
    def __init__(self, reward=0) -> None:
        super().__init__(reward)

class envi:
    def __init__(self,size = [4,4],obstacle_rate = 0.25):
        self.name = 'environment'
        self.size = size
        self.location = []
        self.obstalce_rate = obstacle_rate
        self.total_cells = self.size[0] * self.size[1]

    def generate_new(self):
        self.location = [[cell for i in range(self.size[0])] for j in range(self.size[1])]
        


    def basic_visual(self):
        print(self.location)
    
    def visual_name(self):

        visual_string = ''
        for i in self.location:
            for j in i:
                visual_string += (j.name)
                visual_string += (' ')
            visual_string += ('\n')
        print(visual_string)



if __name__ == '__main__':
    new_map = envi(size = [3,3])
    new_map.generate_new()
    new_map.visual_name()