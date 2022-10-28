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

print(f'\nprediction before training y(5): {forward(5)}\n')

learning_rate = 0.001
epochs = 20

for epoch in range(epochs):

    #forward
    y_pred = forward(X)

    # find loss
    l = loss (Y, y_pred)

    #calculate gradient
    dw = gradient(X,Y,y_pred)

    #update weight
    w -= learning_rate*dw

    print(f'epoch {epoch+1} | w = {w:.3f} | loss = {l:.3f}')

print(f'\nprediction after training y(5): {forward(5)}\n')

"""
output:

prediction before training y(5): 0.5

epoch 1 | w = 1.183 | loss = 114.317
epoch 2 | w = 1.649 | loss = 21.137
epoch 3 | w = 1.849 | loss = 3.908
epoch 4 | w = 1.935 | loss = 0.723
epoch 5 | w = 1.972 | loss = 0.134
epoch 6 | w = 1.988 | loss = 0.025
epoch 7 | w = 1.995 | loss = 0.005
epoch 8 | w = 1.998 | loss = 0.001
epoch 9 | w = 1.999 | loss = 0.000
epoch 10 | w = 2.000 | loss = 0.000
epoch 11 | w = 2.000 | loss = 0.000
epoch 12 | w = 2.000 | loss = 0.000
epoch 13 | w = 2.000 | loss = 0.000
epoch 14 | w = 2.000 | loss = 0.000
epoch 15 | w = 2.000 | loss = 0.000
epoch 16 | w = 2.000 | loss = 0.000
epoch 17 | w = 2.000 | loss = 0.000
epoch 18 | w = 2.000 | loss = 0.000
epoch 19 | w = 2.000 | loss = 0.000
epoch 20 | w = 2.000 | loss = 0.000

prediction after training y(5): 9.999999600648882

"""
