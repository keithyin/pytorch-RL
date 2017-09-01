import gym
import time

env = gym.make('Pendulum-v0')
env.reset()

action = env.action_space.sample()
obs, rew, done, info = env.step(action=action)
print(env.action_space.sample())
print(env.action_space.sample())

