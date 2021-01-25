import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init


class QNetwork(Module):
    """
    Q-network, the inputs is [batch_size, 4, 84, 84]
    """

    def __init__(self, num_actions=2):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        # out [batch_size, 9, 9, 64]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=33280, out_features=512)
        self.out_layer = nn.Linear(in_features=512, out_features=num_actions)
        self.optimizer = None

    def forward(self, inputs):
        net = self.conv1(inputs)
        net = F.relu(net, inplace=True)
        net = self.conv2(net)
        net = F.relu(net, inplace=True)
        net = self.conv3(net)
        net = F.relu(net, inplace=True)
        net = net.view(-1, 33280)
        net = self.fc1(net)
        net = F.relu(net, inplace=True)

        # the output should be the probability of each action
        # using a softmax ??? have no idea
        # just return the logits
        out = self.out_layer(net)
        return out

    def reset_parameters(self, model=None):
        if model is None:
            for layer in self.children():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    init.xavier_normal(layer.weight)
                    init.constant(layer.bias, 0)
            print('the parameters have been initialized.')
        else:
            if not isinstance(model, QNetwork):
                raise ValueError("model must be the same class with this object")
            for param, model_param in zip(self.parameters(), model.parameters()):
                param.data.copy_(model_param.data)
            print('parameters transferred successfully.')

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(self.parameters(), lr=1e-5)
            return self.optimizer
        else:
            return self.optimizer

    def save_model(self, file_path, global_step=None):
        if global_step is None:
            torch.save(self.state_dict(), file_path)
        print('the model has been saved to %s' % file_path)

    def restore_model(self, file_path, global_step=None):
        if global_step is None:
            self.load_state_dict(torch.load(file_path))
        print('the model has been loaded from %s ' % file_path)


def main():
    qnet = QNetwork()
    opt = qnet.get_optimizer()


if __name__ == '__main__':
    main()
