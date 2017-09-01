import gym
from nets.models import Actor
import torch

env = gym.make('Pendulum-v0')
cur_state = env.reset()

actor = Actor(state_dim=3, num_actions=1)
actor.load_state_dict(torch.load('actor.pkl'))
actor.cuda()
