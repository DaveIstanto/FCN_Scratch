#!/bin/python3

import Layers as lys
import numpy as np

#Sample test
x_1 = 255 * np.random.rand(784)
y = np.append([1], 9 * [0])

## Sample test using mnist data
from mnist import MNIST

mndata = MNIST('../dataset')
images, labels = mndata.load_training()

images_array = np.array(images)
labels_array = np.array(labels).reshape((-1, 1))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(labels_array)
labels_array = ohe.transform(labels_array)

x_1 = images_array[100] / 255
y = labels_array[100]

#print(sample_image, sample_label)
# Layers
lin_1 = lys.Linear(784, 300, 10)
relu = lys.Relu()
lin_2  = lys.Linear(300, 10, 10)
sigmoid = lys.Sigmoid()
mse = lys.MSE() 

# n = 1000
# for i in range(n):
#     # Forward pass
#     h1 = lin_1.forward(x_1)
#     act_h1 = relu.forward(h1)
#     h2 = lin_2.forward(act_h1)
#     yhat = sigmoid.forward(h2)
#     loss = mse.forward(yhat, y)

#     # Backward pass
#     g = mse.backward(yhat, y)

#     g = sigmoid.backward(g, yhat)
#     g = lin_2.backward(g, act_h1)

#     g = relu.backward(g, act_h1)
#     g = lin_1.backward(g, x_1)
#     if i == 0 or i == n - 1:
            
#         print(loss)
#         print(yhat)
#         print(f'prediction: {np.argmax(yhat)}')
#         print(f'actual: {np.argmax(y)}')

# Test for multiple training sets
x = [img / 255 for img in images_array[0:1000]]
y = [label for label in labels_array[0:1000]]

# Layers
lin_1 = lys.Linear(784, 300, 1)
relu = lys.Relu()
lin_2  = lys.Linear(300, 10, 1)
sigmoid = lys.Sigmoid()
mse = lys.MSE() 

# Train for 10 epochs
epochs = 10
for epoch in range(epochs):
    right_count = 0 
    epoch_loss = 0
    for sample_index in range(len(x)):
        # Forward Pass
        h1 = lin_1.forward(x[sample_index])
        act_h1 = relu.forward(h1)
        h2 = lin_2.forward(act_h1)
        yhat = sigmoid.forward(h2)
        loss = mse.forward(yhat, y[sample_index])

        # Backward pass
        g = mse.backward(yhat, y[sample_index])
        
        g = sigmoid.backward(g, yhat)
        g = lin_2.backward(g, act_h1)
        
        g = relu.backward(g, act_h1)
        g = lin_1.backward(g, x[sample_index])

        # Stats
        pred = np.argmax(yhat)
        actual = np.argmax(y[sample_index])
        if pred == actual:
            right_count += 1

        epoch_loss += loss

    #if epoch == epochs - 1 or epoch == 0:
    print(f'epoch average loss: {epoch_loss / len(y)}')
    print(f'epoch right count: {right_count}')
    print(f'epoch accuracy: {right_count / len(y)}')

# Test Classifier

x = [img / 255 for img in images_array[1000:1100]]
y = [label for label in labels_array[1000:1100]]

test_right_count = 0
test_loss = 0

for sample_index in range(len(x)):
    # Forward pass
    h1 = lin_1.forward(x[sample_index])
    act_h1 = relu.forward(h1)
    h2 = lin_2.forward(act_h1)
    yhat = sigmoid.forward(h2)
    loss = mse.forward(yhat, y[sample_index])

    # Stats
    pred = np.argmax(yhat)
    actual = np.argmax(y[sample_index])
    if pred == actual:
        test_right_count += 1

    test_loss += test_loss

print(f'test average loss: {test_loss / len(y)}')
print(f'test right count: {test_right_count}')
print(f'test accuracy: {test_right_count / len(y)}')