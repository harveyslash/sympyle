from Nodes import Tensor, Matmul, MSE
import numpy as np

np.random.seed(100)
W = np.random.randn(1, 10)
I = np.random.randn(10, 1)
T = np.random.randn(1, 1)
print(W)
print(I)
print(T)
print("_" * 100)
W_t = Tensor(W)
I_t = Tensor(I)
T_t = Tensor(T)

matmul = Matmul(W_t, I_t)
mse = MSE(matmul, T_t)

# print(mse.forward())

print("printing loss before")
loss = mse.forward()
print(loss)
print("_" * 100)
for i in range(5000):
    loss = mse.forward()
    print(loss)
    grads = matmul.backward(W_t)
    # print(grads.shape)
    # exit()
    W_t.value -= .1 * grads

print("printing loss after")
loss = mse.forward()
print(loss)
print("_" * 100)
print(W_t.value)
# print(I_t.value)
# print(T_t.value)
