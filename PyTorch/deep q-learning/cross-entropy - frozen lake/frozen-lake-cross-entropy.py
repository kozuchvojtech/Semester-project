import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch 
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 15
PERCENTILE = 75
GAMMA = 0.9

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self,env):
        super(DiscreteOneHotWrapper,self).__init__(env)
        assert isinstance(env.observation_space,gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0,1.0,(env.observation_space.n,),dtype=np.float32)
    
    def observation(self,observation):
        res = np.copy(self.observation_space.low)
        res[observation]=1.0
        return res

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
        next_obs, reward, is_done, _ = env.step(action)
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
        if discounted_reward> reward_bound:
            train_obs.extend(map(lambda step:step.observation,example.steps))
            train_act.extend(map(lambda step:step.action,example.steps))
            elite_batch.append(example)
    
    return elite_batch,train_obs,train_act,reward_bound

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env = gym.wrappers.TimeLimit(env,max_episode_steps=100)
    env = DiscreteOneHotWrapper(env)
    
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    network = Network(obs_size,HIDDEN_SIZE,n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = network.parameters(),lr=0.001)
    writer = SummaryWriter(comment='-frozenlake-non_s')
    
    elite_batch = [] 
    for iter_no,batch in enumerate(iterate_batches(env,network,BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda step:step.reward,batch))))
        elite_batch,obs,act,reward_bound= filter_batch(elite_batch+batch,PERCENTILE)
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
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()