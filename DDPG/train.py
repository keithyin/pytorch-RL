import ddpg_pytorch as ddpg
import gym
from nets.models import Actor, Critic


def ddpg_learn_pytorch(env):
    ddpg.learn(
        env,
        Actor_cls=Actor,
        Critic_cls=Critic,
        replay_buffer_size=1000000,
        batch_size=64,
        gamma=0.99,
        learning_starts=500,
        frame_history_len=1
    )
    env.close()


def main():
    env = gym.make('Pendulum-v0')
    ddpg_learn_pytorch(env)


if __name__ == "__main__":
    main()
