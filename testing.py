from Nodes import Tensor, Matmul, MSE, Add
import numpy as np

np.random.seed(100)
W = np.random.randn(1, 10)
I = np.random.randn(10, 1)
T = np.random.randn(1, 1)
S = np.random.randn(1,1)
print(W)
print(I)
print(T)
print("_" * 100)
W_t = Tensor(W)
I_t = Tensor(I)
S_t = Tensor(S)
T_t = Tensor(T)

# dummy = Tensor(np.random.randn(1,10))

matmul = Matmul(W_t, I_t)
add = Add(matmul,S_t)
mse = MSE(add, T_t)

# print(mse.forward())

print("printing loss before")
loss = mse.forward()
print(loss)
print("_" * 100)
for i in range(5000):
    loss = mse.forward()
    print(loss)
    grads = matmul.backward(I_t)
    #print(grads)
    #print(grads.shape)
    # exit()
    I_t.value -= .0001 * grads

print("printing loss after")
loss = mse.forward()
print(loss)
print("_" * 100)
print(W_t.value)
# print(I_t.value)
# print(T_t.value)
