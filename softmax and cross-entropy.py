import math

import torch
import numpy as np
import torch.nn as nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# numpy option
def cross_predicted(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


x = np.array([2.0, 1.0, 0.1])
output = softmax(x)
print("softmax numpy:", output)

for i in range(len(output)):
    output[i] = round(output[i], 2)

print(output)

x = torch.tensor([2.0, 1.0, 0.1])
output2 = torch.softmax(x, dim=0)
print(output2)
Y = np.array([0, 0, 1])

Y_good_pred = np.array([0.7, 0.2, 0.1])
Y_bad_pred = np.array([0.1, 0.3, 0.6])

l1 = cross_predicted(Y, Y_good_pred)
l2 = cross_predicted(Y, Y_bad_pred)

print(f'Cross entropy loss numpy1:{l1:.4f}')
print(f'Cross entropy loss numpy2:{l2:.4f}')

# tensor mode

loss = nn.CrossEntropyLoss()

# 3 samples


Y = torch.tensor([2, 0, 1])
# nsample*nclasses
Y_PRED_GOOD = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_PRED_BAD = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_PRED_GOOD, Y)
l2 = loss(Y_PRED_BAD, Y)
print(l1.item())
print(l2.item())

_, prediction1 = torch.max(Y_PRED_GOOD, 1)
_, prediction2 = torch.max(Y_PRED_BAD, 1)
print(prediction1)
print(prediction2)
