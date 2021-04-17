import numpy as np
from collections import namedtuple
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from simple_value_object import ValueObject

from environment import Maze
from agent import Agent
from visualizer import Game
from visualizer import EpisodeSnapshot
from visualizer import Visualizer

class Map(ValueObject):
    def __init__(self, path, data):
        pass

training_maps = np.array([
    Map(
        'static/map/training_1.json',
        np.matrix([
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', 'C', '%', '%', 'C', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', 'C', '%', '%', 'C', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%']
        ])
    ),
    Map(
        'static/map/training_2.json',
        np.matrix([
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', 'C', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', 'C', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', 'C', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', 'C', '%', '%', '%', '%', '%', '%', '%']
        ])
    ),
    Map(
        'static/map/training_3.json',
        np.matrix([
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', 'C', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', 'C', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', 'C', '%', '%', '%', '%', '%', 'C', '%']
        ])
    )
])

testing = np.matrix([
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', 'C', '%', '.', '.', '%', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
    ['%', '%', '%', '%', '%', 'C', '%', '%', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
    ['%', 'C', '%', '%', '%', '%', '%', '%', 'C', '%'],
    ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%']
])

HIDDEN_SIZE = 256
BATCH_SIZE = 15
GAMMA = 0.98
PERCENTILE = 75
SEED = 1234
LEARNING_RATE = 0.001

env = Maze()
layer_sizes = [env.observations_count, HIDDEN_SIZE, env.actions_count]
agent = Agent(layer_sizes, SEED, LEARNING_RATE)

Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep',field_names = ['observation','action'])

def iterate_batches(maze):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset(maze)

    while True:
        action = agent.sample_action(obs)
        next_obs, reward, is_done = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs,action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward,steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset(maze)

            if len(batch)==BATCH_SIZE:
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

# training

def train(episodeSnapshot):
    for training_map in training_maps:
        elite_batch = []
                
        for iter_no,batch in enumerate(iterate_batches(training_map.data)):

            reward_mean = float(np.mean(list(map(lambda step:step.reward,batch))))
            elite_batch,obs,act,reward_bound = filter_batch(elite_batch+batch,PERCENTILE)

            if not elite_batch:
                continue

            if iter_no % 10 == 0:
                episode = elite_batch[0]
                episode_actions = list(map(lambda step:step.action,episode.steps))
                episodeSnapshot.snapshot(episode_actions, training_map.path)

            loss_value = agent.train(elite_batch, obs, act)

            print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (iter_no, loss_value, reward_mean, reward_bound, len(elite_batch)))

            if reward_mean > 2.5:
                episode = elite_batch[0]
                episode_actions = list(map(lambda step:step.action,episode.steps))
                episodeSnapshot.snapshot(episode_actions, training_map.path)
                break

def main_loop(episodeSnapshot):
    game = Game(episodeSnapshot)

    visualizer = Visualizer(game)
    visualizer.main_loop()
        
if __name__ == '__main__':
    BaseManager.register('EpisodeSnapshot', EpisodeSnapshot)
    manager = BaseManager()
    manager.start()
    episodeSnapshot = manager.EpisodeSnapshot('static/map/training_1.json')

    training_process = Process(target=train, args=(episodeSnapshot,))
    main_loop_process = Process(target=main_loop, args=(episodeSnapshot,))

    training_process.start()
    main_loop_process.start()

    training_process.join(None)
    main_loop_process.join(None)