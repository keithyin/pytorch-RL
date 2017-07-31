from torch.autograd import Variable
import torch

t1 = torch.FloatTensor([1., 2., 3.])
t2 = torch.FloatTensor([2.])
bool__ = Variable(t2) == Variable(torch.FloatTensor([1.]))
print(bool__.data.numpy()[0] == 0.)
