import gym
from nets.models import Actor
import torch
import numpy as np
from torch.autograd import Variable
import time

env = gym.make('Pendulum-v0')
cur_state = env.reset()

actor = Actor(state_dim=3, num_actions=1)
actor.load_state_dict(torch.load('actor.pkl'))
actor.cuda()
warmup = True
while True:
    env.render()
    if warmup:
        time.sleep(1)
        warmup = False
    else:
        time.sleep(1 / 60)
    cur_state = np.expand_dims(cur_state, axis=0)
    cur_state = Variable(torch.FloatTensor(cur_state)).cuda()
    action = actor(cur_state).cpu().data.numpy()[0]
    cur_state, _, done, info = env.step(action)
    if done:
        exit()
