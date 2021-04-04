import gym
from collections import namedtuple
import numpy as np

import networkx as nx
import pylab as plt

import torch
import torch.nn as nn
import torch.optim as optim

from enum import Enum

HIDDEN_SIZE = 128
BATCH_SIZE = 15
PERCENTILE = 75
GAMMA = 0.9
MAZE_SIZE = 5

# maze = np.matrix([
#     ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
#     ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
#     ['%', '%', '.', '.', '%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
#     ['%', '%', '.', '.', '%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
#     ['%', '%', '.', '.', '%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
#     ['%', '%', '%', '%', '%', '%', '%', '%', '.', '.', '.', '.', '%', '%'],
#     ['%', '%', '%', '%', '%', '%', '%', '%', '.', '.', '.', '.', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '%', '%', '%', '.', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '%', '%', '%', '.', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '.', '%', '%', '%', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '.', '%', '%', '%', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '.', '.', '.', '.', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '%', '%', '%', '%', '%', '%'],
#     ['.', '.', '.', '.', '.', '.', '%', '%', '%', '%', '%', '%', '%', 'C']
# ])

maze = np.matrix([
    ['%', '%', '%', '%', '%'],
    ['%', '.', '%', '.', '%'],
    ['%', '.', '%', '.', '%'],
    ['%', '.', '%', '.', '%'],
    ['%', '%', '%', '%', 'C']
])

class Move(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

class MazeEnvironmentWrapper():
    def __init__(self):
        self.init_state = 0
        self.current_state = 0
        self.R = self.get_reward_matrix()
        self.actions_count = 4
        self.observations_count = MAZE_SIZE*MAZE_SIZE
        self.episode_steps_threshold = 100
        self.current_episode_steps = 0

    def get_reward_matrix(self):
        R = np.matrix(np.ones(shape=(MAZE_SIZE,MAZE_SIZE)))
        R *= -1
        
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                if maze[i,j] == '%':
                    R[i,j] = 0
                if maze[i,j] == 'C':
                    R[i,j] = 1
        return R

    def reset(self):
        self.current_state = self.init_state
        self.current_episode_steps = 0
        return self.get_observation(self.current_state)
    
    def step(self, action, training=True):
        self.current_episode_steps += 1
        action = Move(action)

        if action is Move.UP:
            desired_state = self.current_state-MAZE_SIZE
        elif action is Move.RIGHT:
            desired_state = self.current_state+1
        elif action is Move.DOWN:
            desired_state = self.current_state+MAZE_SIZE
        elif action is Move.LEFT:
            desired_state = self.current_state-1
        
        if self.valid_move(desired_state):
            self.current_state = desired_state

        matrix_position = self.get_matrix_position(self.current_state)
        reward = self.R[matrix_position]
        
        next_obs = self.get_observation(self.current_state)

        if training:
            done = maze[matrix_position] == 'C' or maze[matrix_position] == '.' or not self.valid_move(desired_state) or self.current_episode_steps > self.episode_steps_threshold
        else:
            if maze[matrix_position] == 'C':
                self.R[matrix_position] = 0

            done = np.all(self.R <= 0) or maze[matrix_position] == '.' or not self.valid_move(desired_state) or self.current_episode_steps > self.episode_steps_threshold

        return (next_obs, reward, done)

    def valid_move(self, desired_state):
        if desired_state < 0 or desired_state >= MAZE_SIZE*MAZE_SIZE:
            return False
        
        if desired_state == self.current_state - MAZE_SIZE or desired_state == self.current_state + MAZE_SIZE:
            return True
        elif desired_state//MAZE_SIZE == self.current_state//MAZE_SIZE:
            return True

        return False
    
    def get_observation(self, state):
        observation = [0 for _ in range(MAZE_SIZE*MAZE_SIZE)]
        observation[state] = 1

        # for i in range(len(observation)):
        #     if self.R[self.get_matrix_position(i)] > 0:
        #         observation[i] = 2
        return observation
    
    def get_matrix_position(self, state):
        return (state//MAZE_SIZE, state%MAZE_SIZE)

class Network(nn.Module):
    def __init__(self,obs_size,hidden_size,num_actions):
        super(Network,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_actions)
        )
    def forward(self,x):
        return self.network(x)
    
Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep',field_names = ['observation','action'])

def iterate_batches(env,Network,batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob_v = sm(Network(obs_v))
        act_prob = act_prob_v.data.numpy()[0]

        action = np.random.choice(len(act_prob),p=act_prob)
        next_obs, reward, is_done = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs,action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward,steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch)==batch_size:
                yield batch
                batch = []

        obs = next_obs
    
def filter_batch(batch,percentile):
    disc_rewards = list(map(lambda s:s.reward*(GAMMA**len(s.steps)),batch))    
    reward_bound = np.percentile(disc_rewards,percentile)
    
    train_obs = []
    train_act = []
    elite_batch = []
    
    for example,discounted_reward in zip(batch,disc_rewards): 
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step:step.observation,example.steps))
            train_act.extend(map(lambda step:step.action,example.steps))
            elite_batch.append(example)
    
    return elite_batch,train_obs,train_act,reward_bound

if __name__ == "__main__":
    env = MazeEnvironmentWrapper()
    
    network = Network(env.observations_count,HIDDEN_SIZE,env.actions_count)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = network.parameters(),lr=0.001)
    
    elite_batch = []
    for iter_no,batch in enumerate(iterate_batches(env,network,BATCH_SIZE)):

        reward_mean = float(np.mean(list(map(lambda step:step.reward,batch))))
        elite_batch,obs,act,reward_bound = filter_batch(elite_batch+batch,PERCENTILE)

        if not elite_batch:
            continue
        
        obs_v = torch.FloatTensor(obs)
        act_v = torch.LongTensor(act)
        elite_batch = elite_batch[-500:]
        
        optimizer.zero_grad()
        action_scores_v = network(obs_v)
        loss_v = objective(action_scores_v,act_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (iter_no, loss_v.item(), reward_mean, reward_bound, len(elite_batch)))

        if reward_mean >= 0.4:
            obs = env.reset()
            done = False

            steps = [obs]

            while not done:
                action_probabilities = network.forward(torch.FloatTensor([obs]))
                desired_action = np.where(action_probabilities == torch.max(action_probabilities))[1]

                next_obs, reward, done = env.step(desired_action, False)
                obs = next_obs

                steps.append(obs)
            
            print(steps)

            break