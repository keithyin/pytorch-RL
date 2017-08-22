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

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

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
        exploration=exploration_schedule,
        replay_buffer_size=10000,
        batch_size=32,
        gamma=0.99,
        learning_starts=500,
        learning_freq=4,
        frame_history_len=4,
    )
    env.close()


def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    # env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env


def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    atari_learn_pytorch(env, num_timesteps=task.max_timesteps)


if __name__ == "__main__":
    main()
