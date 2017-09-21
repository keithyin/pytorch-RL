import torch
from torch.autograd import Variable
import nets
import time
import numpy as np
import a3c
import utils
from a3c import test_procedure
import os

os.environ['OMP_NUM_THREADS'] = '1'

# def test_procedure(shared_actor, env):
#     num_actions = env.action_space.n
#     local_actor = Actor(num_actions=num_actions)
#     # load parameters from shared models
#
#     # local_actor.load_state_dict(shared_actor.state_dict())
#     while True:
#         replay_buffer = utils.ReplayBuffer(size=4, frame_history_len=4)
#
#         obs = env.reset()
#         rewards = []
#         begin_time = time.time()
#         while True:
#             replay_buffer.store_frame(obs)
#             states = replay_buffer.encode_recent_observation()
#
#             states = np.expand_dims(states, axis=0) / 255.0 - .5
#             logits = local_actor(Variable(torch.FloatTensor(states.astype(np.float32))))
#             action = utils.epsilon_greedy(logits, num_actions=env.action_space.n, epsilon=-1.)
#             obs, reward, done, info = env.step(action)
#             rewards.append(reward)
#             if done:
#                 print("Time:{}, computer:{}, agent:{}".format(time.time() - begin_time,
#                                                               sum(np.array(rewards) == -1),
#                                                               sum(np.array(rewards) == 1)))
#                 break


if __name__ == '__main__':
    from torch import multiprocessing

    env = a3c.get_env()

    print("num actions ", env.action_space.n)

    shared_actor = nets.Actor(num_actions=env.action_space.n)
    # close the env after get the num_action of the game

    shared_critic = nets.Critic()
    # shared_actor_optim = nets.SharedAdam(shared_actor.parameters())
    # shared_critic_optim = nets.SharedAdam(shared_critic.parameters())

    shared_actor.share_memory()
    shared_critic.share_memory()
    # shared_actor_optim.share_memory()
    # shared_critic_optim.share_memory()
    p = multiprocessing.Process(target=test_procedure, args=(shared_actor, a3c.get_env()))

    p.start()
    p.join()
