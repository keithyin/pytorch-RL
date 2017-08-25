import gym
import time

env = gym.make('Pendulum-v0')
env.reset()
for i in range(1000000):
    env.render()
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action=action)
    print(env.observation_space.shape)
    exit()

