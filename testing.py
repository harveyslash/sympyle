from Nodes import Tensor, Matmul, MSE, Relu, Add
import numpy as np

# np.set_printoptions(suppress=True)
np.random.seed(100)

x = np.linspace(1, 100, 100)
print(x)
x = x / 1.
# x = x - np.mean(x)
y = x * x
y = y / 1.0
print(x)
print(y)
# exit()
# x = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
# print(x.shape)
# y = np.array([[0., 0., 0., 0.]]).T
# print(y.shape)
# exit()

I = x.reshape(-1, 1)  # an input with minibatch size 100 and 1 feature
L1 = np.random.uniform(size=(1, 100))  # nn layer with 120 neurons
L1_b = np.random.uniform(size=(100,))

L2 = np.random.uniform(size=(100, 1))  # nn layer with 1 neuron
L2_b = np.random.uniform(size=(1,))

T = y.reshape(-1, 1)

I_t = Tensor(I)
T_t = Tensor(T)

L1_t = Tensor(L1)
L1_bt = Tensor(L1_b)

L2_t = Tensor(L2)
L2_bt = Tensor(L2_b)

matmul1 = Matmul(I_t, L1_t)
add1 = Add(matmul1, L1_bt)
# mse = MSE(add1, T_t)
relu1 = Relu(add1)
#
matmul2 = Matmul(relu1, L2_t)
add2 = Add(matmul2, L2_bt)
relu2 = Relu(add2)
mse = MSE(relu2, T_t)
#
# print("Loss Before training")
# loss = L2_t.forward()
# print(loss.shape)
# # exit()
# print("_" * 80)
lr = .001
for i in range(100):
    # gradsl1 = add1.backward(L1_bt)
    # gradsl1b = add1.backward(L1_bt)

    loss = mse.forward()
    print(i)
    print(loss)

    # gradsl1 = mse.backward(T_t)
    # T_t.value -= lr * gradsl1
    #
    gradsl1 = matmul1.backward(L1_t)
    gradsl1b = add1.backward(L1_bt)
    gradsl2 = matmul2.backward(L2_t)
    gradsl2b = add2.backward(L2_bt)
    # print(gradsl1b.shape)
    # print(L1_bt.value.shape)
    # print(np.average(gradsl1b, axis=0).shape)
    # exit()

    # L1_t.value -= lr * gradsl1
    # L1_bt.value -= lr * np.sum(gradsl1b, axis=0)
    # L2_t.value -= lr * gradsl2
    # L2_bt.value -= lr * np.sum(gradsl2b, axis=0)

    # break
    # print(add2.forward())
    # print(L1_b)
    # grads = add.backward(L1_b)
    # L1_b.value -= lr * grads
#
# x = np.arange(100, 120)
# x = x.reshape(20, 1)
# print(x.shape)
# I_t.value = x.T
print("RESULT")
print(mse.forward().shape)
print(relu2.forward())
