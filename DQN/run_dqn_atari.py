import os.path as osp
import dqn_pytorch as dqn
from gym import wrappers
from nets.qnet_pytorch import QNetwork
from utils.atari_wrappers import *
from utils.dqn_utils import *


def atari_learn_pytorch(env,
                        num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=QNetwork,
        model_ckpt="ckpt/model.pth",
        exploration=exploration_schedule,
        replay_buffer_size=10000,
        batch_size=32,
        gamma=0.99,
        learning_starts=500,
        learning_freq=4,
        frame_history_len=4,
    )
    env.close()


def main():
    env = gym.make("Pong-v0")

    atari_learn_pytorch(env, num_timesteps=1e8)


if __name__ == "__main__":
    main()
