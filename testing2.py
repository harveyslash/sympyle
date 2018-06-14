from Nodes import Tensor, Matmul, MSE
import numpy as np

np.random.seed(100)

I = np.random.randn(5000, 50)  # an input with minibatch size 5 and 50 features
L1 = np.random.randn(100, 50)  # nn layer with 1 neuron
L2 = np.random.randn(1, 100)  # nn layer with 1 neuron

T = np.random.randn(5000, 1)  # target with size 5 ( i.e. list of 5 scalars)

# W_t = Tensor()
# transpose of inputs is taken
I_t = Tensor(I.T)
T_t = Tensor(T.T)

L1_t = Tensor(L1)
L2_t = Tensor(L2)
matmul = Matmul(L1_t, I_t)
matmul2 = Matmul(L2_t, matmul)

mse = MSE(matmul2, T_t)

print("Loss Before training")
loss = mse.forward()
losses = []
print(loss)
print("_" * 80)
for i in range(100000):
    loss = mse.forward()
    losses.append(loss)
    print(loss)
    grads = matmul2.backward(L2_t)
    L2_t.value -= .000000001 * grads

# Plotting
plt.plot(losses)
plt.show()