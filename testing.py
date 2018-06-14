from Nodes import Tensor, Matmul, MSE
import numpy as np

np.random.seed(100)

W = np.random.randn(1, 5000)  # nn layer with 1 neuron
I = np.random.randn(5, 5000)  # an input with minibatch size 5 and 500 features

T = np.random.randn(5, 1)  # target with size 5 ( i.e. list of 5 scalars)

W_t = Tensor(W)
# transpose of inputs is taken
I_t = Tensor(I.T)
T_t = Tensor(T.T)

matmul = Matmul(W_t, I_t)
mse = MSE(matmul, T_t)

print("Loss Before training")
loss = mse.forward()
print(loss)
print("_" * 80)
for i in range(1000):
    loss = mse.forward()
    print(loss)
    grads = mse.backward(T_t)
    T_t.value -= .1 * grads
