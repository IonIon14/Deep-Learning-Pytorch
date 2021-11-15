
import numpy as np

X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)
W = 0.0

#model prediction
def forward(x):
    return W*x

#loss
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

#gradient
#MSE = 1/N*(w*x -y)**2
#dJ/dw=1/N* 2x(w*x - y)
def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    Y_pred = forward(X)
    l = loss(Y,Y_pred)
    dw = gradient(X,Y,Y_pred)

    #update weights
    W-=learning_rate*dw
    if epoch % 1 ==0:
        print(f'epoch {epoch+1}:w = {W:.3f}, loss = {l:.8f}')

print(f'Prediction before training: f(5) = {forward(5):.3f}')