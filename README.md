Author: Jia Yuan

Synopsis: Implementation of reinforcement learning for course ME5406 (Deep Learning in Robotics) in Project 1

Default Settings:
python main.py --method ql --size 4 --new_map False --map_load False --episode 10000 --alpha 0.9 --gamma 0.8 --epsi_init 1.0 --epsi_max 1.0 --epsi_min 0.01 --decay 0.0001 --default True --render False

Generating New map:
python main.py --method ql --size 4 --new_map False --map_load False --episode 10000 --alpha 0.9 --gamma 0.8 --epsi_init 1.0 --epsi_max 1.0 --epsi_min 0.01 --decay 0.0001 --default False --render False

Current Test Settings:
python main.py --method sa --size 4 --new_map False --map_load False --episode 10000 --alpha 0.9 --gamma 0.8 --epsi_init 1.0 --epsi_max 1.0 --epsi_min 0.01 --decay 0.0001 --default True --render False

To Do: 
- Create Environment -Done-
    - State -Done-
    - Payoffs -Done-
    - Visualisation -Done-
- Create Robot -Done-
- Create Metrics for evaluation
- Monte Carlo Methods 
- TD Methods - In progress-
    - SARSA 
    - QL

Important Things to take note of:
- Changes to be made to openai gym import files:
    Location: envs/(your_environment_name,ie. conda)/Lib/site-packages/gym/envs/toy_text/frozen_lake.py

        def update_probability_matrix(row, col, action):
            ......
            reward = 1.0 if newletter == b"G" else -1.0 if newletter == b"H" else 0.0 # This is the new line
            return newstate, reward, terminated
