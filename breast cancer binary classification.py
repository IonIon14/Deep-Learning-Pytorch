import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('data/breast_cancer.csv')
print(df.describe().T)

# rename dataset labels
df = df.rename(columns={'diagnosis': 'label'})
print(df.dtypes)

sns.countplot(x="label", data=df)  # M - malignant B-bening

print("Distribution of data: ", df['label'].value_counts())
y = df['label'].values

print("Labels before encoding are: ", np.unique(y))

labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y)

print("Labels after encoding are:", np.unique(Y))

Y_tensor = torch.from_numpy(Y)
print(Y_tensor)

# Define x and normalize/ scale values
X = df.drop(labels=["label", 'id', "Unnamed: 32"], axis=1)
# df.drop(df.tail(1).index, inplace=True)
print(X.describe().T)  # Need scaling
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)  # scaled

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Shape of training set:", X_train.shape)
print("Shape of testing set:", X_test.shape)

# Deep Learning
model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, Y_train, verbose=1, batch_size=64, epochs=100, validation_data=(X_test, Y_test))
loss = history.history['loss']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label="Training loss")
plt.plot(epochs, val_loss, 'r', label="Validation loss")
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'y', label="Training accuracy")
plt.plot(epochs, val_accuracy, 'r', label="Validation accuracy")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict the Test set
Y_pred = model.predict(X_test)
print("Prediction:", Y_pred)
Y_pred = (Y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

sns.heatmap(cm, annot=True)
