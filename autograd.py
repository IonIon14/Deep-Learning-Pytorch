import torch
x = torch.rand(3,requires_grad=True)

print(x)
y = x.detach()
print(y) # no requires_grad

weights = torch.ones(4,requires_grad=True)

for epoch in range(3  ):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

#
# optimizer = torch.optim.SGD(weights,lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
# print(optimizer)