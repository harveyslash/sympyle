from Nodes import Tensor, Matmul, MSE, Add
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
<<<<<<< HEAD
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
=======
>>>>>>> c00e81c06ef2c0576b0a1adb1ed8ff3f7dda1976

W = np.random.randn(1, 5000)  # nn layer with 1 neuron
I = np.random.randn(5, 5000)  # an input with minibatch size 5 and 500 features

T = np.random.randn(5, 1)  # target with size 5 ( i.e. list of 5 scalars)

W_t = Tensor(W)
# transpose of inputs is taken
I_t = Tensor(I.T)
T_t = Tensor(T.T)

matmul = Matmul(W_t, I_t)
add = Add(matmul,S_t)
mse = MSE(add, T_t)

print("Loss Before training")
loss = mse.forward()
losses = []
print(loss)
print("_" * 80)
for i in range(1000):
    loss = mse.forward()
    losses.append(loss)
    print(loss)
    grads = add.backward(S_t)
    #print(grads)
    #print(grads.shape)
    # exit()
    S_t.value -= .001 * grads

print("printing loss after")
loss = mse.forward()
print(loss)
print("_" * 100)
print(W_t.value)
print(W_t.value.shape)
# print(T_t.value)

# Plotting
print("_"*10)
#print(S_t.value.shape)
plt.plot(losses)
plt.show()

