import torch
import numpy as np

a = torch.FloatTensor([np.array([1., 2.]), np.array([2., 3.])])
c = a.add(1, torch.FloatTensor([3]))
print(c)
