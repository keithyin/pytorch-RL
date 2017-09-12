import a3c
import nets
from torch import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['OMP_NUM_THREADS'] = '1'
import time


def main():
    env = a3c.get_env()

    print("num actions ", env.action_space.n)

    shared_actor = nets.Actor(num_actions=env.action_space.n)
    # close the env after get the num_action of the game

    shared_critic = nets.Critic()
    shared_actor_optim = nets.SharedAdam(shared_actor.parameters())
    shared_critic_optim = nets.SharedAdam(shared_critic.parameters())

    shared_actor.share_memory()
    shared_critic.share_memory()
    shared_actor_optim.share_memory()
    shared_critic_optim.share_memory()

    num_process = 6
    processes = []
    for i in range(num_process):
        processes.append(multiprocessing.Process(target=a3c.learning_thread, args=(shared_actor,
                                                                                   shared_critic,
                                                                                   shared_actor_optim,
                                                                                   shared_critic_optim)))
    for p in processes:
        p.start()

    a3c.test_procedure(shared_actor, env)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
