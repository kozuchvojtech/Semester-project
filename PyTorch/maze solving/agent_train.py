from network import Network
from collections import namedtuple
import numpy as np
import torch as torch
import torch.optim as optim
import torch.nn as nn
from environment import MazeEnvironmentWrapper
import asyncio

HIDDEN_SIZE = 256
BATCH_SIZE = 15
PERCENTILE = 75
GAMMA = 0.98
MAZE_SIZE = 10
SEED = 1234

Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep',field_names = ['observation','action'])

class agent_train():
    def __init__(self, observations_count, actions_count):
        torch.manual_seed(SEED)
        self.network = Network(observations_count,HIDDEN_SIZE,actions_count)
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params = self.network.parameters(),lr=0.001)

    def iterate_batches(self, env):
        batch = []
        episode_reward = 0.0
        episode_steps = []
        obs = env.reset()
        sm = nn.Softmax(dim=1)

        while True:
            obs_v = torch.FloatTensor([obs])

            act_prob_v = sm(self.network(obs_v))
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

                if len(batch)==BATCH_SIZE:
                    yield batch
                    batch = []

            obs = next_obs
    
    def filter_batch(self,batch,percentile):
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

    def train(self, env, player, callback):    
        elite_batch = []
        
        for iter_no,batch in enumerate(self.iterate_batches(env)):

            reward_mean = float(np.mean(list(map(lambda step:step.reward,batch))))
            elite_batch,obs,act,reward_bound = self.filter_batch(elite_batch+batch,PERCENTILE)

            if not elite_batch:
                continue
            
            if iter_no % 10 == 0:
                episode = elite_batch[0]
                episode_actions = list(map(lambda step:step.action,episode.steps))
                callback(player, episode_actions)
            
            obs_v = torch.FloatTensor(obs)
            act_v = torch.LongTensor(act)
            elite_batch = elite_batch[-500:]
            
            self.optimizer.zero_grad()
            action_scores_v = self.network(obs_v)
            loss_v = self.objective(action_scores_v,act_v)
            loss_v.backward()
            self.optimizer.step()

            print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (iter_no, loss_v.item(), reward_mean, reward_bound, len(elite_batch)))

            if reward_mean > 2.5:
                episode = elite_batch[0]
                episode_actions = list(map(lambda step:step.action,episode.steps))
                callback(player, episode_actions)
                print(episode_actions)
                break

    def save_trained_model(self, model_path="model/maze-solving-dqn.pth"):
        torch.save(self.network.state_dict(), model_path)

    def load_pretrained_model(self, model_path="model/maze-solving-dqn.pth"):
        self.network.load_state_dict(torch.load(model_path))

    def test(self, env):
        obs = env.reset()
        done = False
        steps = [obs]

        while not done:
            action_probabilities = self.network.forward(torch.FloatTensor([obs]))
            desired_action = np.where(action_probabilities == torch.max(action_probabilities))[1]

            next_obs, reward, done = env.step(desired_action, False)
            obs = next_obs

            steps.append(obs)
        
        print(steps)