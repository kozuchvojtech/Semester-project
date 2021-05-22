import numpy as np
import math as math
from move import Move

class Maze():
    def __init__(self, actions_count=4, observations_count=12, episode_threshold=50):        
        self.actions_count = actions_count
        self.observations_count = observations_count
        self.init_state = 0
        self.current_episode_steps = 0
        self.current_state = self.init_state
        self.episode_steps_threshold = episode_threshold

    def get_reward_matrix(self):
        """A reward matrix is populated for internal purposes only. Zero value means there's a road at a given position. Non-negative value then means that a coin is there. Otherwise it's a field the agent isn't allowed to step on.

        Returns:
            matrix: R-matrix with values populated
        """
        R = np.matrix(np.ones(shape=(self.maze.shape[0],self.maze.shape[0])))
        R *= -1
        
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[0]):
                if self.maze[i,j] == '%':
                    R[i,j] = 0
                if self.maze[i,j] == 'C':
                    R[i,j] = 1
        return R
    
    def update_reward_matrix(self):
        self.R = self.get_reward_matrix()

    def reset(self, maze, hero_position, on_coin_grabbed=None):
        self.maze = maze
        self.init_state = self.get_state(hero_position)

        self.current_state = self.init_state
        self.previous_state = self.init_state
        self.current_episode_steps = 0
        self.update_reward_matrix()
        self.on_coin_grabbed = on_coin_grabbed

        return self.get_observation(self.current_state)
    
    def step(self, action):
        """The state of the agent is resolved based on the action required. There are several ways of how the desired state would be considered a final move. Either the agent found all coins succesfully or an invalid move was made.

        Args:
            action (enum): an enum value representing the action requested

        Returns:
            tuple: combination of the next observation based on the current state, reward for the action taken and information whether the game ends or not
        """
        self.current_episode_steps += 1
        action = Move(action)

        if action is Move.UP:
            desired_state = self.current_state-self.maze.shape[0]
        elif action is Move.RIGHT:
            desired_state = self.current_state+1
        elif action is Move.DOWN:
            desired_state = self.current_state+self.maze.shape[0]
        elif action is Move.LEFT:
            desired_state = self.current_state-1
        
        if self.valid_move(desired_state):
            self.previous_state = self.current_state
            self.current_state = desired_state

        matrix_position = self.get_matrix_position(self.current_state)
        reward = self.R[matrix_position]
        
        if self.maze[matrix_position] == 'C':
            self.R[matrix_position] = 0

            if self.on_coin_grabbed:
                self.on_coin_grabbed(matrix_position)

        next_obs = self.get_observation(self.current_state)

        done = np.all(self.R <= 0) or \
               self.maze[matrix_position] == '.' or  \
               not self.valid_move(desired_state) or \
               (self.episode_steps_threshold and self.current_episode_steps > self.episode_steps_threshold)
               
        return (next_obs, reward, done)

    def valid_move(self, desired_state):
        if desired_state < 0 or desired_state >= self.maze.shape[0]*self.maze.shape[0]:
            return False
        
        if desired_state == self.current_state - self.maze.shape[0] or desired_state == self.current_state + self.maze.shape[0]:
            return True
        elif desired_state//self.maze.shape[0] == self.current_state//self.maze.shape[0]:
            return True

        return False
    
    def get_observation(self, state):
        """This method generates observation based on the state received. The observation consists of information about the surrounding area of the agent, location of the nearest coin and previous agent's location.

        Args:
            state (float): current state of the agent

        Returns:
            array: current observation
        """
        current_position = self.get_matrix_position(state)        
        previous_position = self.get_matrix_position(self.previous_state)

        obstacle_positions = []
        coin_positions = []
        nearest_coin_distance = None

        for i in range(self.maze.shape[0]*self.maze.shape[0]):
            matrix_position = self.get_matrix_position(i)
            if self.R[matrix_position] > 0:
                coin_positions.append(matrix_position)
            elif self.R[matrix_position] < 0:
                obstacle_positions.append(matrix_position)

        if coin_positions:
            lowest_distance = np.min([self.euclidean_distance(current_position,coin) for coin in coin_positions])
            nearest_coin_index = np.where([self.euclidean_distance(current_position,coin) == lowest_distance for coin in coin_positions])[0]

            if nearest_coin_index.shape[0] > 1:
                nearest_coin_index = int(np.random.choice(nearest_coin_index, size=1))
            else:
                nearest_coin_index = int(nearest_coin_index)

            nearest_coin = coin_positions[nearest_coin_index]

            nearest_coin_max_distance = np.amax(np.array([
                abs(current_position[0] - nearest_coin[0]),      
                abs(current_position[1] - nearest_coin[1])
            ]))

            nearest_coin_max_distance = nearest_coin_max_distance if nearest_coin_max_distance > 0 else 1
            nearest_coin_distance = (((abs(current_position[0] - nearest_coin[0])) / nearest_coin_max_distance), ((abs(current_position[1] - nearest_coin[1])) / nearest_coin_max_distance))

        observation = [
            np.any([obs == (current_position[0]-1,current_position[1]) for obs in obstacle_positions]) or current_position[0]-1 < 0,
            np.any([obs == (current_position[0],current_position[1]+1) for obs in obstacle_positions]) or current_position[1]+1 >= self.maze.shape[0],
            np.any([obs == (current_position[0]+1,current_position[1]) for obs in obstacle_positions]) or current_position[0]+1 >= self.maze.shape[0],
            np.any([obs == (current_position[0],current_position[1]-1) for obs in obstacle_positions]) or current_position[1]-1 < 0,
            current_position[0] > previous_position[0],
            current_position[1] < previous_position[1],
            current_position[0] < previous_position[0],
            current_position[1] > previous_position[1],
            nearest_coin_distance[0] if nearest_coin_distance and current_position[0] > nearest_coin[0] else 0,      
            nearest_coin_distance[1] if nearest_coin_distance and current_position[1] < nearest_coin[1] else 0,
            nearest_coin_distance[0] if nearest_coin_distance and current_position[0] < nearest_coin[0] else 0,
            nearest_coin_distance[1] if nearest_coin_distance and current_position[1] > nearest_coin[1] else 0
        ]

        for i in range(len(observation)-4):
            if observation[i]:
                observation[i]=1
            else:
                observation[i]=0

        return observation
    
    def get_matrix_position(self, state):
        return (state//self.maze.shape[0], state%self.maze.shape[0])
    
    def get_state(self, matrix_position):
        return matrix_position[0]*self.maze.shape[0] + matrix_position[1]

    def euclidean_distance(self, position_a, position_b):
        return math.sqrt(math.pow(position_a[1] - position_b[1], 2) + math.pow(position_a[0] - position_b[0], 2))