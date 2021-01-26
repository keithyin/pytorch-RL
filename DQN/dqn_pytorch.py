import itertools
from collections import namedtuple
import torch
import gym.spaces
from utils.dqn_utils import *
import os
from utils import common_algorithm

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

cuda_available = torch.cuda.is_available()


def learn(env,
          q_func,
          model_ckpt,
          exploration=LinearSchedule(1000000, 0.1),
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    Q_net = q_func(num_actions=num_actions)
    target_net = q_func(num_actions=num_actions)
    if os.path.exists(model_ckpt):
        Q_net.restore_model(model_ckpt)
        target_net.restore_model(model_ckpt)

    if cuda_available:
        Q_net.cuda()
        target_net.cuda()

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    last_obs = env.reset()
    action_one_hot = torch.FloatTensor(batch_size, num_actions)
    reward_track = []
    episode_num = 0
    for t in itertools.count():
        # env.render()
        ### 1. Check stopping criterion
        # if stopping_criterion is not None and stopping_criterion(env, t):
        #     break
        # if stopping_criterion is not None:
        #     break
        # #### behavior policy epsilon-greedy ##############
        # get the current state                            #
        # choose an action from the epsilon-greedy         #
        # take an action                                   #
        # get the reward and observe the next observation  #
        ####################################################
        idx = replay_buffer.store_frame(np.mean(last_obs, axis=2, keepdims=True))
        cur_state = replay_buffer.encode_recent_observation() / 255.0
        cur_state = torch.from_numpy(np.expand_dims(cur_state, axis=0)).float()
        if cuda_available:
            cur_state = cur_state.cuda()

        with torch.no_grad():
            logits = Q_net(cur_state)
        #
        action = common_algorithm.epsilon_greedy(logits=logits, num_actions=num_actions,
                                                 epsilon=exploration.value(t))
        observation, reward, done, info = env.step(action)
        replay_buffer.store_effect(idx=idx, action=action, reward=reward, done=done)
        reward_track.append(reward)
        if done:
            reward_track = np.array(reward_track)
            win = np.sum(reward_track == 1)
            lose = np.sum(reward_track == -1)
            print("episode %d , computer %d VS %d agent" % (episode_num, lose, win))
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
            # ndarray -> torch.autograd.Variable
            obs_batch = torch.FloatTensor(obs_batch / 255.0)
            # act_batch = torch.IntTensor(act_batch)
            rew_batch = torch.FloatTensor(rew_batch)
            next_obs_batch = torch.FloatTensor(next_obs_batch / 255.0)
            not_done_mask = torch.FloatTensor(1 - done_mask)

            if cuda_available:
                obs_batch = obs_batch.cuda()
                next_obs_batch = next_obs_batch.cuda()
                not_done_mask = not_done_mask.cuda()
                rew_batch = rew_batch.cuda()
            # calculate the target,
            target_next_state_value = target_net(inputs=next_obs_batch)
            target_next_state_value.detach_()
            target_next_state_value = target_next_state_value.max(1)[0]
            target_next_state_value.data.mul_(gamma)

            # if done, using the reward as the target
            target_next_state_value.data.mul_(not_done_mask)

            # target_next_state_value:shape [batch_size]
            target_next_state_value.data.add_(rew_batch)
            Q_value = Q_net(obs_batch)
            # choose the value corresponding to action
            common_algorithm.one_hot(act_batch, out_tensor=action_one_hot)
            action_mask = action_one_hot.bool()
            if cuda_available:
                action_mask = action_mask.cuda()
            selected_value = torch.masked_select(Q_value, mask=action_mask)
            loss = torch.mean(torch.pow(target_next_state_value - selected_value, 2))
            Q_net.get_optimizer().zero_grad()
            loss.backward()
            Q_net.get_optimizer().step()

            # update the target network
            if t % learning_freq == 0:
                target_net.load_state_dict(Q_net.state_dict())

            if t % (learning_freq * 1000) == 0:
                target_net.save_model(model_ckpt)


def main():
    pass


if __name__ == '__main__':
    main()
