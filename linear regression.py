import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0)prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# convert to tensors
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape
# 1)model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2)loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3)training loop
num_epochs = 100

for epoch in range(num_epochs):
    y_pred = model(X)
    l = criterion(y_pred, Y)
    # backward pass
    l.backward()
    # update
    optimizer.step()

    optimizer.zero_grad()
    if ((epoch + 1) % 10 == 0):
        print(f'epoch:{epoch + 1}, loss={l.item()}:.4f')
# forward pass and loss

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()
