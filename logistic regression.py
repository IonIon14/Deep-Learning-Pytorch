import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # for train test separation
import matplotlib.pyplot as plt

# 0)prepare data
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# scale features

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1)  # only one row and
Y_test = Y_test.view(Y_test.shape[0], 1)


# 1)model
# f = wx +y sigmoid at the end

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(n_features)

# 2)loss and optimizer
learning_rate = 0.2
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3)training loop

nr_epochs = 200

for epoch in range(nr_epochs):
    y_pred = model(X_train)
    l = criterion(y_pred, Y_train)
    # backward pass
    l.backward()
    # update
    optimizer.step()

    optimizer.zero_grad()
    if ((epoch + 1) % 10 == 0):
        print(f'epoch:{epoch + 1}, loss={l.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')
