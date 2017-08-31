import torch
from torch.nn import Module
import torch.nn as nn
from torch import optim
from torch.nn import init
from collections import OrderedDict
import torch.nn.functional as F


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
        self.bn1 = nn.BatchNorm1d(num_features=400, affine=False)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        self.bn2 = nn.BatchNorm1d(num_features=300, affine=False)
        self.output = nn.Linear(in_features=300, out_features=num_actions)
        self.reset_parameters()

    def forward(self, states):
        net = self.fc1(states)
        net = self.bn1(net)
        net = F.relu(net, inplace=True)
        net = self.fc2(net)
        net = self.bn2(net)
        net = F.relu(net, inplace=True)
        net = self.output(net)
        action = 2 * F.tanh(net)
        return action

    def reset_parameters(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.xavier_normal(layer.weight)
                init.constant(layer.bias, 0)
        print("the parameter has been initialized...")

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
        self.bn1 = nn.BatchNorm1d(num_features=400, affine=False)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        self.fc2_action = nn.Linear(in_features=num_actions, out_features=300)
        self.bn2 = nn.BatchNorm1d(num_features=300, affine=False)
        self.output = nn.Linear(in_features=300, out_features=1)
        self.reset_parameters()

    def forward(self, states, actions):
        """
        forward process: get the Q(s,a)
        :param states: [batch_size, state_feature]
        :param actions: [batch_size, action]
        :return: Q(s,a)
        """
        net = self.fc1(states)
        net = self.bn1(net)
        net = F.relu(net, inplace=True)
        net = self.fc2(net)
        action_net = self.fc2_action(actions)
        net = net + action_net
        net = self.bn2(net)
        net = F.relu(net, inplace=True)
        value = self.output(net)
        return value

    def reset_parameters(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.xavier_normal(layer.weight)
                init.constant(layer.bias, 0)
        print("the parameter has been initialized...")

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


def main():
    actor = Actor(state_dim=3, num_actions=1)
    print(len(list(actor.parameters())))


if __name__ == '__main__':
    main()
