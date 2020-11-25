##
import torch.nn.functional as F
import torch

a = torch.empty(10, 3, 5, 5)
b = torch.empty(10, 3, 5, 5)
##
from collections import OrderedDict

lst = OrderedDict()

##
lst['conv1'] = 3
lst['conv2'] = 4

##
import numpy as np
a = np.array([1,2,3])

##
b = np.asarray(a)

##
b[0] = 4

##

