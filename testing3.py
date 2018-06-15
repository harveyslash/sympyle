from Nodes import Tensor, Matmul, MSE, Relu, Add
import numpy as np

np.random.seed(100)

x = np.arange(100)
y = x
print(x)
print(y)
# exit()

I = x.reshape(100, 1)  # an input with minibatch size 100 and 1 feature

L1 = np.random.randn(120, 1)  # nn layer with 120 neurons
L1_b = np.random.randn(120, 100)

L2 = np.random.randn(1, 120)  # nn layer with 1 neuron
L2_b = np.random.randn(1, 100)

T = y.reshape(100, 1)

I_t = Tensor(I.T)
T_t = Tensor(T.T)

L1_t = Tensor(L1)
L1_bt = Tensor(L1_b)

L2_t = Tensor(L2)
L2_bt = Tensor(L2_b)

matmul1 = Matmul(L1_t, I_t)
add1 = Add(matmul1, L1_bt)
relu1 = Relu(add1)

matmul2 = Matmul(L2_t, relu1)
add2 = Add(matmul2, L2_bt)
mse = MSE(add2, T_t)

print("Loss Before training")
loss = L2_t.forward()
print(loss.shape)
# exit()
print("_" * 80)
lr = .001
for i in range(1000000):
    loss = mse.forward()
    if i == 0 or i%1000:
        print(loss)
    gradsl1 = matmul1.backward(L1_t)
    gradsl1b = add1.backward(L1_bt)

    gradsl2 = matmul2.backward(L2_t)
    gradsl2b = add2.backward(L2_bt)

    L1_t.value -= lr * gradsl1
    L1_bt.value -= lr * gradsl1b
    L2_t.value -= lr * gradsl2
    L2_bt.value -= lr * gradsl2b

    # print(L1_b)
    # grads = add.backward(L1_b)
    # L1_b.value -= lr * grads