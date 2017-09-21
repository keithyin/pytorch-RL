import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Actor(Module):
    def __init__(self, num_actions=2):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        # out [batch_size, 9, 9, 64]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=10 * 10 * 64, out_features=512)
        self.out_layer = nn.Linear(in_features=512, out_features=num_actions)

        self.optimizer = None
        self.reset_parameters()

    def forward(self, inputs):
        net = self.conv1(inputs)
        net = F.relu(net, inplace=True)
        net = self.conv2(net)
        net = F.relu(net, inplace=True)
        net = self.conv3(net)
        net = F.relu(net, inplace=True)

        net = net.view(-1, 10 * 10 * 64)
        net = self.fc1(net)
        net = F.relu(net, inplace=True)
        logits = self.out_layer(net)

        return logits

    def reset_parameters(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.xavier_normal(layer.weight)
                init.constant(layer.bias, 0)
        print('the parameters have been initialized.')


class Critic(Module):
    """
    Q-network, the inputs is [batch_size, 4, 84, 84]
    """

    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        # out [batch_size, 9, 9, 64]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=10 * 10 * 64, out_features=512)
        self.out_layer = nn.Linear(in_features=512, out_features=1)
        self.optimizer = None
        self.reset_parameters()

    def forward(self, inputs):
        net = self.conv1(inputs)
        net = F.relu(net, inplace=True)
        net = self.conv2(net)
        net = F.relu(net, inplace=True)
        net = self.conv3(net)
        net = F.relu(net, inplace=True)
        net = net.view(-1, 10 * 10 * 64)
        net = self.fc1(net)
        net = F.relu(net, inplace=True)
        out = self.out_layer(net)
        return out

    def reset_parameters(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.xavier_normal(layer.weight)
                init.constant(layer.bias, 0)
        print('the parameters have been initialized.')

    def save_model(self, file_path, global_step=None):
        if global_step is None:
            torch.save(self.state_dict(), file_path)
        print('the model has been saved to %s' % file_path)

    def restore_model(self, file_path, global_step=None):
        if global_step is None:
            self.load_state_dict(torch.load(file_path))
        print('the model has been loaded from %s ' % file_path)


def main():
    critic = Critic()
    opt = critic.get_optimizer()


if __name__ == '__main__':
    main()
