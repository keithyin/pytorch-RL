from torch import nn
import torch
from torch.autograd import Variable
from utils.dqn_utils import *
import nets
import utils
import itertools
import time


def get_env():
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    env_id = task.env_id

    env = gym.make(env_id)

    env.seed(0)

    expt_dir = '/tmp/hw3_vid_dir2/'
    # env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = utils.wrap_deepmind(env)
    return env


def set_shared_grad(model, shared_model):
    if isinstance(shared_model, type(model)):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if param.grad is None:
                continue
            shared_param._grad = param.grad


def learning_thread(shared_actor,
                    shared_critic,
                    shared_actor_optim,
                    shared_critic_optim,
                    exploration=LinearSchedule(1000000, 0.1),
                    gamma=0.99,
                    frame_history_len=4):
    ####
    # 1. build a local model
    # 2. synchronize the shared model parameters and local model
    # 3. choose an action based on observation
    # 4. take an action and get the reward and the next observation
    # 5. calculate the target, and accumulate the gradient
    # 6. update the global model
    ####

    # prepare environment
    env = get_env()
    obs = env.reset()
    num_actions = env.action_space.n
    # prepare local model
    local_actor = nets.Actor(num_actions=num_actions)
    local_critic = nets.Critic()

    # criterion
    criterion = nn.MSELoss(size_average=False)

    # load parameters from shared models
    local_actor.load_state_dict(shared_actor.state_dict())
    local_critic.load_state_dict(shared_critic.state_dict())

    replay_buffer = utils.ReplayBuffer(size=4, frame_history_len=frame_history_len)

    #
    idx = replay_buffer.store_frame(obs)

    num_n_steps = 4
    for i in itertools.count():
        states = []
        actions = []
        next_states = []
        dones = []
        rewards = []
        for i in range(num_n_steps):
            replay_buffer.store_frame(obs)
            state = replay_buffer.encode_recent_observation()
            state = np.expand_dims(state, axis=0) / 255.0 - .5

            state = Variable(torch.from_numpy(state.astype(np.float32)), volatile=True)
            logits = local_actor(state)
            action = utils.epsilon_greedy(logits, num_actions=num_actions, epsilon=exploration(i))
            next_obs, reward, done, info = env.step(action)

            replay_buffer.store_frame(next_obs)
            # store the states for get the gradients
            states.append(state)
            actions.append(action)
            dones.append(done)
            rewards.append(reward)
            next_states.append(replay_buffer.encode_recent_observation())

            if done:
                break
        # compute targets and compute the critic's gradient
        # from numpy to torch.Variable
        cur_states = np.array(states) / 255.0 - .5
        cur_states = Variable(torch.FloatTensor(cur_states.astype(np.float32)))
        next_states = np.array(next_states) / 255.0 - .5
        next_states = Variable(torch.FloatTensor(next_states.astype(np.float32)), volatile=True)
        not_done_mask = torch.FloatTensor(1 - np.array(dones).astype(dtype=np.float32)).view_(-1, 1)
        rewards = torch.FloatTensor(np.array(rewards).astype(np.float32)).view_(-1, 1)
        values = local_critic(next_states)
        targets = values.data.mul_(not_done_mask).mul_(gamma)
        targets = targets.add_(rewards)

        # compute the gradient of the critic network
        values = local_critic(cur_states)
        loss = criterion(values, Variable(targets))
        local_critic.zero_grad()
        loss.backward()

        # compute the gradient of the actor network
        # first : compute the head gradient

        head_gradient = -(targets - values) * utils.one_hot(actions, depth=num_actions)
        print(head_gradient)
        actions_prob = local_actor(cur_states)

        local_actor.zero_grad()
        actions_prob.backward(head_gradient)

        set_shared_grad(local_actor, shared_actor)
        set_shared_grad(local_critic, shared_critic)
        shared_actor_optim.step()
        shared_critic_optim.step()


def test_procedure(shared_actor, env):
    num_actions = env.action_space.n
    local_actor = nets.Actor(num_actions=num_actions)
    # load parameters from shared models
    while True:
        local_actor.load_state_dict(shared_actor.state_dict())

        replay_buffer = utils.ReplayBuffer(size=4, frame_history_len=4)

        obs = env.reset()
        rewards = []
        begin_time = time.time()
        while True:
            replay_buffer.store_frame(obs)
            states = replay_buffer.encode_recent_observation()
            states = np.expand_dims(states, axis=0) / 255.0 - .5
            logits = local_actor(Variable(torch.FloatTensor(states.astype(np.float32))))
            action = utils.epsilon_greedy(logits, num_actions=env.action_space.n, epsilon=-1.)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                print("Time:{}, computer:{}, agent:{}".format(time.time() - begin_time,
                                                              sum(np.array(rewards) == -1),
                                                              sum(np.array(rewards) == 1)))


def main():
    pass


if __name__ == '__main__':
    main()
