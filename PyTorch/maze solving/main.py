import numpy as np
from collections import namedtuple
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from simple_value_object import ValueObject
import plotly.graph_objects as go
import random
import pandas as pd
import pygame

import plotly.graph_objects as go

from environment import Maze
from agent import Agent
from visualizer import Game
from visualizer import EpisodeSnapshot
from visualizer import Visualizer
from move import Move

class Map(ValueObject):
    def __init__(self, path, data):
        pass

training_maps = np.array([
    Map(
        'static/map/training.json',
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
    )
])

testing_map = Map(
    'static/map/testing.json',
    np.matrix([
        ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
        ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
        ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%']
    ])
)

HIDDEN_SIZE = 256
BATCH_SIZE = 20
GAMMA = 0.98
PERCENTILE = 25
SEED = 1234
LEARNING_RATE = 0.003
EPISODES_THRESHOLD = 200
DESIRED_REWARD = 3.75
MAP_ITERATIONS = 256

env = Maze()
layer_sizes = [env.observations_count, HIDDEN_SIZE, env.actions_count]
agent = Agent(layer_sizes, SEED, LEARNING_RATE)

Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep',field_names = ['observation','action'])

def iterate_batches(maze, hero_position, epsilon):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset(maze, hero_position)

    while True:
        action = agent.sample_action(obs, epsilon)
        next_obs, reward, is_done = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs,action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward,steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset(maze, hero_position)

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
        if discounted_reward > reward_bound or (reward_bound > 2 and discounted_reward >= reward_bound):
            train_obs.extend(map(lambda step:step.observation,example.steps))
            train_act.extend(map(lambda step:step.action,example.steps))
            elite_batch.append(example)
        
    return elite_batch,train_obs,train_act,reward_bound

def generate_coins(maze, count=4, grabbed_coin_position=None):
    if grabbed_coin_position:
        maze[grabbed_coin_position] = '%'
    else:
        maze[maze == 'C'] = '%'

    road_positions = zip(np.where(maze == '%')[0], np.where(maze == '%')[1])
    roads = list(map(tuple, road_positions))

    coin_positions = random.sample(roads, count)

    for coin_position in coin_positions:
        maze[coin_position] = 'C'

    return coin_positions

def generate_hero(maze):
    road_positions = zip(np.where(maze == '%')[0], np.where(maze == '%')[1])
    roads = list(map(tuple, road_positions))

    return random.choice(roads)

def train(episodeSnapshot):
    iterations = []
    loss_func_values = []
    reward_mean_values = []
    epsilon_values = []

    iterations_count = 0
    last_snapshot_iteration = 0
    
    initial_iteration = True
    epsilon = 1

    for training_map in training_maps:            
        for map_iter in range(MAP_ITERATIONS):

            if initial_iteration:
                initial_iteration = False
                coin_positions = zip(np.where(training_map.data == 'C')[0], np.where(training_map.data == 'C')[1])
                coins = list(map(tuple, coin_positions))
                hero_position = [0,0]
            else:
                coins = generate_coins(training_map.data)
                hero_position = generate_hero(training_map.data)
            
            elite_batch = []        

            for iter_no,batch in enumerate(iterate_batches(training_map.data, hero_position, epsilon)):

                reward_mean = float(np.mean(list(map(lambda step:step.reward,batch))))
                elite_batch,obs,act,_ = filter_batch(elite_batch+batch,PERCENTILE)

                if not elite_batch:
                    if iter_no > EPISODES_THRESHOLD:
                        break
                    else:
                        continue

                if iter_no == 0 or abs(last_snapshot_iteration-iter_no) > 25:
                    episode = elite_batch[0]
                    episode_actions = list(map(lambda step:step.action,episode.steps))
                    episodeSnapshot.snapshot(hero_position, episode_actions, episode.reward, training_map.path, coins)
                    last_snapshot_iteration = iter_no

                loss_value = agent.train(elite_batch, obs, act)
                elite_batch = elite_batch[-100:]
                
                iterations_count += 1
                iterations.append(iterations_count)
                loss_func_values.append(loss_value)
                reward_mean_values.append(reward_mean)
                epsilon_values.append(epsilon)

                print("%d/%d: loss=%.3f, reward_mean=%.3f, epsilon=%.3f" % (map_iter, iter_no, loss_value, reward_mean, epsilon))
                                
                if reward_mean > DESIRED_REWARD or iter_no > EPISODES_THRESHOLD:
                    episode = elite_batch[0]
                    episode_actions = list(map(lambda step:step.action,episode.steps))
                    episodeSnapshot.snapshot(hero_position, episode_actions, episode.reward, training_map.path, coins)

                    if epsilon > 0.1:
                        epsilon -= 0.001
                    break

                if epsilon > 0.1:
                    epsilon -= 0.001
                

    print('done training!')

    fig = go.Figure()
    fig.update_layout(template='plotly_white')

    fig.add_trace(go.Scatter(
        x=iterations, 
        y=reward_mean_values,
        mode='lines',
        name='reward mean',
        line=dict(
            color='#82B366',
            width=1.5
        )))

    fig.add_trace(go.Scatter(
        x=iterations, 
        y=loss_func_values,
        mode='lines',
        name='loss value',
        line=dict(
            color='#B85450',
            width=1.5
        )))

    fig.add_trace(go.Scatter(
        x=iterations, 
        y=epsilon_values,
        mode='lines',
        name='epsilon value', 
        line=dict(
            color='#6C8EBF',
            width=1.5
        )))

    fig.write_image("training-process.pdf")
    agent.save_trained_model()

class Tester():
    def __init__(self):             
        coins = generate_coins(testing_map.data)
        self.hero_position = generate_hero(testing_map.data)
        episodeSnapshot = EpisodeSnapshot('static/map/testing.json', coins, self.hero_position)

        self.game = Game(episodeSnapshot, True)
        self.env = Maze(episode_threshold=None)

    def on_coin_grabbed(self, maze_position):        
        coin = generate_coins(testing_map.data, count=1, grabbed_coin_position=maze_position)[0]

        self.game.append_coin(coin)
        self.env.update_reward_matrix()

    def test(self):
        agent.load_pretrained_model()

        obs = self.env.reset(testing_map.data, self.hero_position, self.on_coin_grabbed)
        done = False
        actions = []
        reward_sum = 0

        visualization_done = False

        while not visualization_done:
            if not done:
                action = agent.choose_action(obs)
                next_obs, reward, done = self.env.step(action)
                reward_sum += reward

                obs = next_obs
                actions.append(action)

            if actions:
                visualization_done, _ = self.game.play(Move(actions.pop(0)))
            else:
                visualization_done, _ = self.game.play()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    visualization_done = True
                    done = True

        self.game.gameOver(reward_sum)

def main_loop(episodeSnapshot):
    game = Game(episodeSnapshot)

    visualizer = Visualizer(game)
    visualizer.main_loop()

def process_training():
    BaseManager.register('EpisodeSnapshot', EpisodeSnapshot)
    manager = BaseManager()
    manager.start()
    episodeSnapshot = manager.EpisodeSnapshot('static/map/training.json')

    training_process = Process(target=train, args=(episodeSnapshot,))
    main_loop_process = Process(target=main_loop, args=(episodeSnapshot,))

    training_process.start()
    main_loop_process.start()

    training_process.join(None)
    main_loop_process.join(None)

def process_testing():
    Tester().test()
        
if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.intro(process_training, process_testing)