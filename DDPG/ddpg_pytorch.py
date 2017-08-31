"""
this is an implementation of DDPG algorithm
"""

import itertools
from collections import namedtuple
import torch
import gym.spaces
from torch.autograd import Variable
from utils.dqn_utils import *
from utils import common_algorithm

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

cuda_available = torch.cuda.is_available()


def get_target_value_double_dqn(target_net, Q_net, out_one_hot, next_obs_batch):
    """
    compute the double dqn's target value
    :param target_net: Critic object:  target net for approximating target value
    :param Q_net: Critic object: Q-net for get the action
    :param out_one_hot: FloatTensor
    :param next_obs_batch: Variable
    :return:
    """
    target_next_state_value = Q_net(inputs=next_obs_batch)
    target_next_state_value.detach_()
    target_next_action = target_next_state_value.max(1)[1]
    if cuda_available:
        target_next_action = target_next_action.cpu()
    common_algorithm.one_hot(target_next_action.data.numpy(), out_tensor=out_one_hot)
    target_next_state_value = target_net(next_obs_batch)
    target_next_state_value.detach_()
    action_mask = Variable(out_one_hot.type(new_type=torch.ByteTensor))
    if cuda_available:
        action_mask = action_mask.cuda()
    selected_value = torch.masked_select(target_next_state_value, mask=action_mask)

    return selected_value


def get_target_value_dqn(target_net, out_one_hot, next_obs_batch):
    """
    compute the dqn's target value
    :param target_net: Critic Object
    :param out_one_hot: Float Tensor
    :param next_obs_batch: Variable
    :return: target value

    """
    # TODO: ... fill this function
    pass


def get_target_value_critic(target_critic_net, target_actor_net, next_obs_batch):
    """
    get the target value
    :param target_critic_net: Critic object
    :param target_actor_net: Actor object
    :param next_obs_batch: Variable, next states
    :return: value
    """
    # TODO: try to using net with eval procedure, or set the next_obs_batch.volatile=True to speedup
    actions = target_actor_net(next_obs_batch)
    target_value = target_critic_net(states=next_obs_batch, actions=actions)
    target_value.detach_()
    return target_value


def learn(env,
          Actor_cls,
          Critic_cls,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          frame_history_len=1):
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Box

    ###############
    # BUILD MODEL #
    ###############

    actor_net = Actor_cls(state_dim=3, num_actions=1)
    critic_net = Critic_cls(state_dim=3, num_actions=1)
    target_actor_net = Actor_cls(state_dim=3, num_actions=1)
    target_critic_net = Critic_cls(state_dim=3, num_actions=1)

    if cuda_available:
        actor_net.cuda()
        target_actor_net.cuda()
        critic_net.cuda()
        target_critic_net.cuda()

    # construct the replay buffer, frame history len is 1 if use low-dimensional features
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    last_obs = env.reset()
    reward_track = []
    episode_num = 0
    for t in itertools.count():
        # #### behavior policy epsilon-greedy ##############
        # get the current state                            #
        # choose an action from the epsilon-greedy         #
        # take an action                                   #
        # get the reward and observe the next observation  #
        ####################################################
        idx = replay_buffer.store_frame(last_obs)
        cur_state = replay_buffer.encode_recent_observation()
        if cuda_available:
            cur_state = Variable(torch.FloatTensor(np.expand_dims(cur_state, axis=0)), volatile=True).cuda()
        else:
            cur_state = Variable(torch.FloatTensor(np.expand_dims(cur_state, axis=0)), volatile=True)

        # choose the action at the current state
        # TODO: add Ornstein-Uhlenbeck process
        action = actor_net(cur_state)
        action_numpy = action.cpu().data.numpy()[0]
        action_numpy = action_numpy + np.random.normal(loc=0., scale=.1)

        observation, reward, done, info = env.step(action_numpy)
        replay_buffer.store_effect(idx=idx, action=action_numpy, reward=reward, done=done)
        reward_track.append(reward)
        if done:
            reward_track = np.array(reward_track)
            episode_rew = np.sum(reward_track)
            print("episode %d , reward %d" % (episode_num, episode_rew))
            episode_num += 1
            reward_track = []
            observation = env.reset()

        last_obs = observation

        # ####### experience replay ################
        # sample mini-batch from replay buffer     #
        # calculate the target                     #
        # train the online net                     #
        # parameter synchronization every n step   #
        ############################################
        if (t > learning_starts and
                replay_buffer.can_sample(batch_size)):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
                replay_buffer.sample(batch_size=batch_size)

            rew_batch = np.expand_dims(rew_batch, axis=-1)
            # ndarray -> torch.autograd.Variable
            obs_batch = Variable(torch.FloatTensor(obs_batch))
            act_batch = Variable(torch.FloatTensor(np.expand_dims(act_batch, axis=-1)))
            rew_batch = torch.FloatTensor(rew_batch)
            next_obs_batch = Variable(torch.FloatTensor(next_obs_batch))
            not_done_mask = torch.FloatTensor(np.expand_dims(1 - done_mask, axis=-1))

            if cuda_available:
                obs_batch = obs_batch.cuda()
                next_obs_batch = next_obs_batch.cuda()
                not_done_mask = not_done_mask.cuda()
                rew_batch = rew_batch.cuda()
                act_batch = act_batch.cuda()

            # ###################DDPG##################
            # 1. get target value, update critic-net  #
            # 2. update actor-net                     #
            # 3. update target-nets..                 #
            ###########################################
            # step1
            target_next_state_value = get_target_value_critic(target_critic_net=target_critic_net,
                                                              target_actor_net=target_actor_net,
                                                              next_obs_batch=next_obs_batch)
            target_next_state_value.data.mul_(gamma)
            # if done, using the reward as the target
            target_next_state_value.data.mul_(not_done_mask)
            # target_next_state_value:shape [batch_size]
            target_next_state_value.data.add_(rew_batch)

            Q_value = critic_net(states=obs_batch, actions=act_batch)

            assert Q_value.size() == target_next_state_value.size()

            loss = torch.mean(torch.pow(target_next_state_value - Q_value, 2))
            critic_net.get_optimizer().zero_grad()
            loss.backward()
            critic_net.get_optimizer().step()
            # step1: Done#####################################################

            # step2:
            value = torch.mean(torch.neg(critic_net(states=obs_batch, actions=actor_net(obs_batch))))
            actor_net.get_optimizer().zero_grad()
            value.backward()
            actor_net.get_optimizer().step()
            # step2: Done#####################################################

            # step3:
            target_actor_net.moving_average_update(state_dict=actor_net.state_dict(), decay=.99)
            target_critic_net.moving_average_update(state_dict=critic_net.state_dict(), decay=.99)

            if t % 500 == 0:
                torch.save(target_critic_net.state_dict(), 'critic.pkl')
                torch.save(target_actor_net.state_dict(), 'actor.pkl')
                print('num iteration: ', t, ", the nets' parameters have been saved")


def main():
    pass


if __name__ == '__main__':
    main()
