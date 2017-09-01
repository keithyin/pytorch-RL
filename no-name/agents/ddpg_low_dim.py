import numpy as np
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from collections import OrderedDict

from utils.replay_memory import ReplayMemory, Transition

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def init_fanin(tensor):
    fanin = tensor.size(1)
    v = 1.0 / np.sqrt(fanin)
    init.uniform(tensor, -v, v)


# class Actor(nn.Module):
#     def __init__(self, num_feature, num_action):
#         """
#         Initialize a Actor for low dimensional environment.
#             num_feature: number of features of input.
#             num_action: number of available actions in the environment.
#         """
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(num_feature, 400)
#         init_fanin(self.fc1.weight)
#         self.fc2 = nn.Linear(400, 300)
#         init_fanin(self.fc2.weight)
#         self.fc3 = nn.Linear(300, num_action)
#         init.uniform(self.fc3.weight, -3e-3, 3e-3)
#         init.uniform(self.fc3.bias, -3e-3, 3e-3)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.tanh(self.fc3(x))
#         return x


# class Critic(nn.Module):
#     def __init__(self, num_feature, num_action):
#         """
#         Initialize a Critic for low dimensional environment.
#             num_feature: number of features of input.
#
#         """
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(num_feature, 400)
#         init_fanin(self.fc1.weight)
#         # Actions were not included until the 2nd hidden layer of Q.
#         self.fc2 = nn.Linear(400 + num_action, 300)
#         init_fanin(self.fc2.weight)
#         self.fc3 = nn.Linear(300, 1)
#         init.uniform(self.fc3.weight, -3e-3, 3e-3)
#         init.uniform(self.fc3.bias, -3e-3, 3e-3)
#
#     def forward(self, states, actions):
#         x = F.relu(self.fc1(states))
#         # Actions were not included until the 2nd hidden layer of Q.
#         x = torch.cat((x, actions), 1)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class Actor(Module):
    def __init__(self, state_dim, num_actions):
        """
        for CartPole state_dim=4 (position_of_cart, velocity_of_cart, angle_of_pole, rotation_rate_of_pole)
        :param state_dim: int
        :param num_actions: int ,if None, action space is continuous
        :return Nothing
        """
        super(Actor, self).__init__()
        self.optimizer = None
        self.fc1 = nn.Linear(in_features=state_dim, out_features=400)
        init_fanin(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        init_fanin(self.fc2.weight)
        self.output = nn.Linear(in_features=300, out_features=num_actions)
        init.uniform(self.output.weight, -3e-3, 3e-3)
        init.uniform(self.output.bias, -3e-3, 3e-3)

    def forward(self, states):
        net = self.fc1(states)
        net = F.relu(net, inplace=True)
        net = self.fc2(net)
        net = F.relu(net, inplace=True)
        net = self.output(net)
        action = F.tanh(net)
        return action

    def save_model(self, file_path, global_step=None):
        if global_step is None:
            torch.save(self.state_dict(), file_path)
        print('the model has been saved to %s' % file_path)

    def restore_model(self, file_path, global_step=None):
        if global_step is None:
            self.load_state_dict(torch.load(file_path))
        print('the model has been loaded from %s ' % file_path)

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = optim.Adam(params=self.parameters(), lr=1e-4)
            return self.optimizer
        else:
            return self.optimizer

    def moving_average_update(self, state_dict, decay=.99):
        # decay v = decay*v + (1-decay)*new_v
        assert isinstance(state_dict, OrderedDict)
        for k, v in self.state_dict().items():
            v.mul_(decay)
            v.add_((1 - decay) * state_dict[k])


class Critic(Module):
    def __init__(self, state_dim, num_actions):
        """
        for CartPole state_dim=4 (position_of_cart, velocity_of_cart, angle_of_pole, rotation_rate_of_pole)
        :param state_dim: int
        :param num_actions: int ,if None, action space is continuous
        :return Nothing
        """
        super(Critic, self).__init__()
        self.optimizer = None
        self.fc1 = nn.Linear(in_features=state_dim, out_features=400)
        init_fanin(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        init_fanin(self.fc2.weight)
        self.fc2_action = nn.Linear(in_features=num_actions, out_features=300)
        init_fanin(self.fc2_action.weight)
        self.output = nn.Linear(in_features=300, out_features=1)
        init.uniform(self.output.weight, -3e-3, 3e-3)
        init.uniform(self.output.bias, -3e-3, 3e-3)

    def forward(self, states, actions):
        """
        forward process: get the Q(s,a)
        :param states: [batch_size, state_feature]
        :param actions: [batch_size, action]
        :return: Q(s,a)
        """
        net = self.fc1(states)
        net = F.relu(net, inplace=True)
        net = self.fc2(net)
        action_net = self.fc2_action(actions)
        net = net + action_net
        net = F.relu(net, inplace=True)
        value = self.output(net)
        return value

    def save_model(self, file_path, global_step=None):
        if global_step is None:
            torch.save(self.state_dict(), file_path)
        print('the model has been saved to %s' % file_path)

    def restore_model(self, file_path, global_step=None):
        if global_step is None:
            self.load_state_dict(torch.load(file_path))
        print('the model has been loaded from %s ' % file_path)

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = optim.Adam(params=self.parameters(), lr=1e-3)
            return self.optimizer
        else:
            return self.optimizer

    def moving_average_update(self, state_dict, decay=.99):
        # decay v = decay*v + (1-decay)*new_v
        assert isinstance(state_dict, OrderedDict)
        for k, v in self.state_dict().items():
            v.mul_(decay)
            v.add_((1 - decay) * state_dict[k])


class DDPG():
    """
    The Deep Deterministic Policy Gradient (DDPG) Agent
    Parameters
    ----------
        actor_optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate and other
            parameters for the optimizer
        critic_optimizer_spec: OptimizerSpec
        num_feature: int
            The number of features of the environmental state
        num_action: int
            The number of available actions that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        tau: float
            The update rate that target networks slowly track the learned networks.
    """

    def __init__(self,
                 actor_optimizer_spec,
                 critic_optimizer_spec,
                 num_feature,
                 num_action,
                 replay_memory_size=1000000,
                 batch_size=64,
                 tau=0.001):
        ###############
        # BUILD MODEL #
        ###############
        self.num_feature = num_feature
        self.num_action = num_action
        self.batch_size = batch_size
        self.tau = tau
        # Construct actor and critic
        self.actor = Actor(num_feature, num_action).type(dtype)
        self.target_actor = Actor(num_feature, num_action).type(dtype)
        self.critic = Critic(num_feature, num_action).type(dtype)
        self.target_critic = Critic(num_feature, num_action).type(dtype)
        # Construct the optimizers for actor and critic
        self.actor_optimizer = actor_optimizer_spec.constructor(self.actor.parameters(),
                                                                **actor_optimizer_spec.kwargs)
        self.critic_optimizer = critic_optimizer_spec.constructor(self.critic.parameters(),
                                                                  **critic_optimizer_spec.kwargs)
        # Construct the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size)

    def select_action(self, state):
        state = torch.from_numpy(state).type(dtype).unsqueeze(0)
        action = self.actor(Variable(state, volatile=True)).data.cpu()[0, 0]
        return action

    def update(self, gamma=1.0):
        if len(self.replay_memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_mask = \
            self.replay_memory.sample(self.batch_size)
        state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
        action_batch = Variable(torch.from_numpy(action_batch).type(dtype)).unsqueeze(1)
        reward_batch = Variable(torch.from_numpy(reward_batch).type(dtype))
        next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

        ### Critic ###
        # Compute current Q value, critic takes state and action choosen
        current_Q_values = self.critic(state_batch, action_batch)
        # Compute next Q value based on which action target actor would choose
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        target_Q_values = get_target_value_critic(self.target_critic, self.target_actor, next_state_batch)
        target_Q_values = torch.squeeze(target_Q_values)

        target_Q_values.data.mul_(gamma)
        # if done, using the reward as the target
        target_Q_values.data.mul_(not_done_mask.data)
        # target_next_state_value:shape [batch_size]
        target_Q_values.data.add_(reward_batch.data)

        # Compute Bellman error (using Huber loss)
        critic_loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
        # critic_loss = torch.mean(torch.pow(target_Q_values - current_Q_values, 2))
        # Optimize the critic
        self.critic.get_optimizer().zero_grad()
        critic_loss.backward()
        self.critic.get_optimizer().step()

        ### Actor ###
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        # Optimize the actor
        self.actor.get_optimizer().zero_grad()
        actor_loss.backward()
        self.actor.get_optimizer().step()

        # Update the target networks
        self.target_actor.moving_average_update(self.actor.state_dict(), decay=1 - self.tau)
        self.target_critic.moving_average_update(self.critic.state_dict(), decay=1 - self.tau)


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


def update_(actor_net, critic_net, target_actor_net, target_critic_net, replay_buffer, batch_size, gamma):
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
        replay_buffer.sample(batch_size=batch_size)

    rew_batch = np.expand_dims(rew_batch, axis=-1)
    # ndarray -> torch.autograd.Variable
    obs_batch = Variable(torch.FloatTensor(obs_batch)).cuda()
    act_batch = Variable(torch.FloatTensor(np.expand_dims(act_batch, axis=-1))).cuda()
    rew_batch = torch.FloatTensor(rew_batch).cuda()
    next_obs_batch = Variable(torch.FloatTensor(next_obs_batch)).cuda()
    not_done_mask = torch.FloatTensor(np.expand_dims(1 - done_mask, axis=-1).astype(np.float32)).cuda()

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

    # using huber loss, MSE loss doesn't work .... why
    loss = F.smooth_l1_loss(Q_value, target_next_state_value)
    critic_net.get_optimizer().zero_grad()
    loss.backward()
    critic_net.get_optimizer().step()
    # step1: Done#####################################################

    # step2:
    value = -torch.mean(critic_net(states=obs_batch, actions=actor_net(obs_batch)))
    actor_net.get_optimizer().zero_grad()
    value.backward()
    actor_net.get_optimizer().step()
    # step2: Done#####################################################

    # step3:
    target_actor_net.moving_average_update(state_dict=actor_net.state_dict(), decay=.999)
    target_critic_net.moving_average_update(state_dict=critic_net.state_dict(), decay=.999)
