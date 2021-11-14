import torch
import numpy as np

a = np.ones(5)

print(a)

b = torch.from_numpy(a)
print(b)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    # creates tensor on GPU
    device = torch.device("cuda")
    x= torch.ones(5,device=device)
    y= torch.ones(5)
    y = y.to(device)
    z = x + y #performed on CPU
    z = z.to("cpu")







