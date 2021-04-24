import numpy as np
from collections import namedtuple
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from simple_value_object import ValueObject
import plotly.graph_objects as go

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
    ),
    Map(
        'static/map/training_4.json',
        np.matrix([
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', 'C', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', 'C'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', 'C', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', 'C', '%', '%']
        ])
    ),
    Map(
        'static/map/training_5.json',
        np.matrix([
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', 'C'],
            ['%', '%', '%', '%', 'C', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', 'C'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', 'C', '%', '%', '%', '%', '%', '%', '%']
        ])
    ),
    Map(
        'static/map/training_6.json',
        np.matrix([
            ['C', '%', '%', '%', '%', '%', '%', '%', '%', 'C'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', 'C', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', 'C']
        ])
    ),
    Map(
        'static/map/training_7.json',
        np.matrix([
            ['C', '%', '%', '%', '%', '%', '%', '%', '%', 'C'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', 'C', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['C', '%', '%', '%', '%', '%', '%', '%', '%', '%']
        ])
    ),
    Map(
        'static/map/training_8.json',
        np.matrix([
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', 'C', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', 'C'],
            ['%', '%', '.', '.', '%', '%', '.', '.', '%', '%'],
            ['%', '%', '%', 'C', '%', '%', '%', '%', '%', '%'],
            ['%', '%', '%', '%', '%', '%', '%', 'C', '%', '%']
        ])
    )
])

testing_map = Map(
    'static/map/testing.json',
    np.matrix([
        ['%', '%', '%', '%', '%', 'C', '%', '%', '%', '%'],
        ['%', '%', '%', '%', '%', '%', '%', '%', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', 'C'],
        ['%', 'C', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', 'C'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', '%', '.', '.', '.', '.', '.', '.', '%', '%'],
        ['%', 'C', '%', '%', '%', '%', '%', '%', '%', '%'],
        ['%', '%', '%', '%', '%', 'C', '%', '%', '%', '%']
    ])
)

HIDDEN_SIZE = 256
BATCH_SIZE = 15
GAMMA = 0.98
PERCENTILE = 25
SEED = 1234
LEARNING_RATE = 0.003
EPISODES_THRESHOLD = 250
DESIRED_REWARD = 3.75

env = Maze()
layer_sizes = [env.observations_count, HIDDEN_SIZE, env.actions_count]
agent = Agent(layer_sizes, SEED, LEARNING_RATE)

Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep',field_names = ['observation','action'])

def iterate_batches(maze, epsilon):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset(maze)

    while True:
        action = agent.sample_action(obs, epsilon)
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

def train(episodeSnapshot):
    iterations = []
    loss_func_values = []
    reward_mean_values = []

    iterations_count = 0
    last_snapshot_iteration = 0

    for training_map in training_maps:
        epsilon = 1
        elite_batch = []
                
        for iter_no,batch in enumerate(iterate_batches(training_map.data, epsilon)):

            reward_mean = float(np.mean(list(map(lambda step:step.reward,batch))))
            elite_batch,obs,act,reward_bound = filter_batch(elite_batch+batch,PERCENTILE)

            if not elite_batch:
                continue

            if iterations_count % 20 == 0 or abs(last_snapshot_iteration-iter_no) > 50:
                episode = elite_batch[0]
                episode_actions = list(map(lambda step:step.action,episode.steps))
                episodeSnapshot.snapshot(episode_actions, episode.reward, training_map.path)
                last_snapshot_iteration = iter_no

            loss_value = agent.train(elite_batch, obs, act)

            iterations_count += 1
            iterations.append(iterations_count)
            loss_func_values.append(loss_value)
            reward_mean_values.append(reward_mean)

            print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, epsilon=%.3f" % (iter_no, loss_value, reward_mean, reward_bound, epsilon))
            
            if reward_mean > DESIRED_REWARD or iter_no > EPISODES_THRESHOLD:
                episode = elite_batch[0]
                episode_actions = list(map(lambda step:step.action,episode.steps))
                episodeSnapshot.snapshot(episode_actions, episode.reward, training_map.path)
                break

            if epsilon > 0.01:
                epsilon -= 0.001
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=loss_func_values, name='loss function'))
    fig.add_trace(go.Scatter(x=iterations, y=reward_mean_values, name='reward mean'))
    
    fig.write_image("training_loss.png")
    agent.save_trained_model()

def test(episodeSnapshot):
    agent.load_pretrained_model()

    obs = env.reset(testing_map.data)
    done = False
    actions = []
    reward_sum = 0

    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, done = env.step(action)
        reward_sum += reward

        obs = next_obs
        actions.append(action)

    episodeSnapshot.snapshot(actions, reward_sum, testing_map.path)

def main_loop(episodeSnapshot, map_path):
    game = Game(episodeSnapshot, map_path)

    visualizer = Visualizer(game)
    visualizer.main_loop()

def process_training():
    BaseManager.register('EpisodeSnapshot', EpisodeSnapshot)
    manager = BaseManager()
    manager.start()
    episodeSnapshot = manager.EpisodeSnapshot('static/map/training_1.json')

    training_process = Process(target=train, args=(episodeSnapshot,))
    main_loop_process = Process(target=main_loop, args=(episodeSnapshot,'static/map/default.json'))

    training_process.start()
    main_loop_process.start()

    training_process.join(None)
    main_loop_process.join(None)

def process_testing():
    BaseManager.register('EpisodeSnapshot', EpisodeSnapshot)
    manager = BaseManager()
    manager.start()
    episodeSnapshot_testing = manager.EpisodeSnapshot('static/map/testing.json')

    testing_process = Process(target=test, args=(episodeSnapshot_testing,))
    main_loop_process = Process(target=main_loop, args=(episodeSnapshot_testing,'static/map/testing_map.json'))

    testing_process.start()
    main_loop_process.start()

    testing_process.join(None)
    main_loop_process.join(None)
        
if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.intro(process_training, process_testing)