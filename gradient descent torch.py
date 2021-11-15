
#1) Design model (input size, output size, forward pass)
#2) Constract loss and optimizer
#3) Training loop
#   -forward pass : compute prediction
#   -backward pass : compute gradients
#   -update weights

import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)


X_test = torch.tensor([5],dtype=torch.float32)
n_samples , n_features = X.shape

input_size = n_features
output_size = n_features
print(n_samples,n_features)

model =  nn.Linear(input_size,output_size)

#loss

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#Training

learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) # stochastic gradient decent

for epoch in range(n_iters):
    Y_pred = model(X)
    l = loss(Y,Y_pred)

    #gradient = backward pass dl/dw
    l.backward()
    #update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 ==0:
        [W,b] = model.parameters()
        print(f'epoch {epoch+1}:w = {W[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction before training: f(5) = {model(X_test):.3f}')