'''
1) build model
2) make loss and optimizer
3) training loop:
    - forward pass
    - backward pass
    - update weight

'''
import torch
import torch.nn as nn


X = torch.arange(1,10, dtype= torch.float32).reshape(9,1)
Y = 2 * X

X_test = torch.tensor([5], dtype=torch.float32)
# print(Y)

w = torch.tensor(0.1, dtype= torch.float32, requires_grad= True) #initial weight

# model forward pass
# def forward(x):
#     return w*x
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)



# print(forward(10))

#loss = MSE
# y = 2*x 
# 1/N*(y_pred -y)**2



print(f'\nprediction before training y(5): {model (X_test).item()}\n')

learning_rate = 0.001
epochs = 120

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(epochs):

    #forward
    y_pred = model(X)

    # find loss
    l = loss (Y, y_pred)

    #calculate gradient
    l.backward()


    optimizer.step()

    
    optimizer.zero_grad()

    [w, b] = model.parameters()
    print(f'epoch {epoch+1} | w = {w.data.item():.3f} | b = {b.data.item():.3f}| loss = {l:.3f}')

print(f'\nprediction after training y(5): {model(X_test).item( )}\n')

"""

prediction before training y(5): 0.5

epoch 1 | w = 0.220 | loss = 114.317
epoch 2 | w = 0.333 | loss = 100.295
epoch 3 | w = 0.439 | loss = 87.993
epoch 4 | w = 0.538 | loss = 77.200
epoch 5 | w = 0.630 | loss = 67.731
epoch 6 | w = 0.717 | loss = 59.424
epoch 7 | w = 0.798 | loss = 52.135
epoch 8 | w = 0.874 | loss = 45.740
epoch 9 | w = 0.946 | loss = 40.130
epoch 10 | w = 1.012 | loss = 35.208
epoch 11 | w = 1.075 | loss = 30.889
epoch 12 | w = 1.133 | loss = 27.101
epoch 13 | w = 1.188 | loss = 23.777
epoch 14 | w = 1.240 | loss = 20.860
epoch 15 | w = 1.288 | loss = 18.302
epoch 16 | w = 1.333 | loss = 16.057
epoch 17 | w = 1.375 | loss = 14.087
epoch 18 | w = 1.415 | loss = 12.360
epoch 19 | w = 1.452 | loss = 10.844
epoch 20 | w = 1.487 | loss = 9.514
epoch 21 | w = 1.519 | loss = 8.347
epoch 22 | w = 1.550 | loss = 7.323
epoch 23 | w = 1.578 | loss = 6.425
epoch 24 | w = 1.605 | loss = 5.637
epoch 25 | w = 1.630 | loss = 4.945
epoch 26 | w = 1.653 | loss = 4.339
epoch 27 | w = 1.675 | loss = 3.807
epoch 28 | w = 1.696 | loss = 3.340
epoch 29 | w = 1.715 | loss = 2.930
epoch 30 | w = 1.733 | loss = 2.571
epoch 31 | w = 1.750 | loss = 2.255
epoch 32 | w = 1.766 | loss = 1.979
epoch 33 | w = 1.781 | loss = 1.736
epoch 34 | w = 1.795 | loss = 1.523
epoch 35 | w = 1.808 | loss = 1.336
epoch 36 | w = 1.820 | loss = 1.172
epoch 37 | w = 1.831 | loss = 1.029
epoch 38 | w = 1.842 | loss = 0.902
epoch 39 | w = 1.852 | loss = 0.792
epoch 40 | w = 1.861 | loss = 0.695
epoch 41 | w = 1.870 | loss = 0.609
epoch 42 | w = 1.878 | loss = 0.535
epoch 43 | w = 1.886 | loss = 0.469
epoch 44 | w = 1.893 | loss = 0.412
epoch 45 | w = 1.900 | loss = 0.361
epoch 46 | w = 1.906 | loss = 0.317
epoch 47 | w = 1.912 | loss = 0.278
epoch 48 | w = 1.918 | loss = 0.244
epoch 49 | w = 1.923 | loss = 0.214
epoch 50 | w = 1.928 | loss = 0.188
epoch 51 | w = 1.932 | loss = 0.165
epoch 52 | w = 1.937 | loss = 0.144
epoch 53 | w = 1.941 | loss = 0.127
epoch 54 | w = 1.944 | loss = 0.111
epoch 55 | w = 1.948 | loss = 0.098
epoch 56 | w = 1.951 | loss = 0.086
epoch 57 | w = 1.954 | loss = 0.075
epoch 58 | w = 1.957 | loss = 0.066
epoch 59 | w = 1.960 | loss = 0.058
epoch 60 | w = 1.963 | loss = 0.051
epoch 61 | w = 1.965 | loss = 0.044
epoch 62 | w = 1.967 | loss = 0.039
epoch 63 | w = 1.969 | loss = 0.034
epoch 64 | w = 1.971 | loss = 0.030
epoch 65 | w = 1.973 | loss = 0.026
epoch 66 | w = 1.975 | loss = 0.023
epoch 67 | w = 1.976 | loss = 0.020
epoch 68 | w = 1.978 | loss = 0.018
epoch 69 | w = 1.979 | loss = 0.016
epoch 70 | w = 1.981 | loss = 0.014
epoch 71 | w = 1.982 | loss = 0.012
epoch 72 | w = 1.983 | loss = 0.011
epoch 73 | w = 1.984 | loss = 0.009
epoch 74 | w = 1.985 | loss = 0.008
epoch 75 | w = 1.986 | loss = 0.007
epoch 76 | w = 1.987 | loss = 0.006
epoch 77 | w = 1.988 | loss = 0.005
epoch 78 | w = 1.988 | loss = 0.005
epoch 79 | w = 1.989 | loss = 0.004
epoch 80 | w = 1.990 | loss = 0.004
epoch 81 | w = 1.991 | loss = 0.003
epoch 82 | w = 1.991 | loss = 0.003
epoch 83 | w = 1.992 | loss = 0.003
epoch 84 | w = 1.992 | loss = 0.002
epoch 85 | w = 1.993 | loss = 0.002
epoch 86 | w = 1.993 | loss = 0.002
epoch 87 | w = 1.994 | loss = 0.001
epoch 88 | w = 1.994 | loss = 0.001
epoch 89 | w = 1.994 | loss = 0.001
epoch 90 | w = 1.995 | loss = 0.001
epoch 91 | w = 1.995 | loss = 0.001
epoch 92 | w = 1.995 | loss = 0.001
epoch 93 | w = 1.996 | loss = 0.001
epoch 94 | w = 1.996 | loss = 0.001
epoch 95 | w = 1.996 | loss = 0.001
epoch 96 | w = 1.996 | loss = 0.000
epoch 97 | w = 1.997 | loss = 0.000
epoch 98 | w = 1.997 | loss = 0.000
epoch 99 | w = 1.997 | loss = 0.000
epoch 100 | w = 1.997 | loss = 0.000

prediction after training y(5): 9.98631477355957
"""

