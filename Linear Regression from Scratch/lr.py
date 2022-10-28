import numpy as np

# suppose we have this function for parameter estimation: y = 2*x

X = np.arange(1,10, dtype= np.float32)
Y = 2 * X

# print(Y)

w = 0.1 #initial weight

# model forward pass
def forward(x):
    return w*x

# print(forward(10))

#loss = MSE
# y = 2*x 
# 1/N*(y_pred -y)**2
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

#loss = 1/N*(y_pred -y)**2 -> we want to find dl/dw
# loss = 1/N *(w.x-y)**2 -> dl/dw = 2/N*(w.x-y)(x)
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

print(f'prediction before training y(5): {forward(5)}')

learning_rate = 0.001
epochs = 20

for epoch in range(epochs):

    y_pred = forward(X)

    l = loss (Y, y_pred)

    dw = gradient(X,Y,y_pred)

    w -= learning_rate*dw

    print(f'epoch {epoch+1} | w = {w:.3f} | loss = {l:.3f}')

print(f'prediction after training y(5): {forward(5)}')
