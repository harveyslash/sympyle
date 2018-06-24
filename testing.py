from Nodes import Tensor, Matmul, MSE, Relu, Add
import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(100)

X = np.linspace(-np.pi, np.pi, 8000).reshape(-1, 1).astype(np.float32)
X_copy = X.copy()
pure = .0 * X + .8
pure[100:500] = .4 * X[100:500] + .8
pure[500:] = -.5 * X[500:] + .8
pure[500:600] = -.05 * X[500:600] + .8
pure = np.cos(X)

Y = pure + np.random.randn(*X.shape) * .07
Y = Y.astype(np.float32)

I = X.reshape(-1, 1)  # an input with minibatch size 100 and 1 feature
# L1 = np.random.uniform(size=(1, 100))  # nn layer with 120 neurons
# L1_b = np.random.uniform(size=(1, 100))
#
# L2 = np.random.uniform(size=(100, 1))  # nn layer with 1 neuron
# L2_b = np.random.uniform(size=(1, 1))

L1 = np.random.randn(1, 100).astype(np.float32)
L1_b = np.random.randn(1, 100).astype(np.float32)

L2 = np.random.randn(100, 1).astype(np.float32)
L2_b = np.random.randn(1, 1).astype(np.float32)

T = Y.reshape(-1, 1)

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
# relu2 = Relu(add2)
mse = MSE(add2, T_t)
#
# print("Loss Before training")
# loss = L2_t.forward()
# print(loss.shape)
# # exit()
# print("_" * 80)
lr = .001
print("-" * 100)
for i in range(1):
    # gradsl1 = add1.backward(L1_bt)
    # gradsl1b = add1.backward(L1_bt)

    # loss = mse.backward()
    # print(i)
    #
    # print(loss)
    # exit()

    # gradsl1 = mse.backward(T_t)
    # T_t.value -= lr * gradsl1
    #
    # gradsl1 = matmul1.backward(L1_t)
    # gradsl1b = add1.backward(L1_bt)
    # gradsl2 = matmul1.backward(L1_t)
    # gradsl2b = matmul1.backward(L1_t)
    # print(gradsl2b)
    # gradsl2b = matmul1.backward(L1_t)
    # print(gradsl2b)
    # gradsl2b = matmul1.backward(L1_t)
    # print(gradsl2b)
    print(mse.forward())
    mse.backward()
    # print(matmul2.backward_val)
    print(L1_bt.backward_val.sum(axis=0))

    # print(np.sum(gradsl2b, axis=0, keepdims=True))
    # mse.clear_caches()
    exit()
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
print("-" * 100)
#
# x = np.arange(100, 120)
# x = x.reshape(20, 1)
# print(x.shape)
# I_t.value = x.T
print("RESULT")
print(mse.forward().shape)
print(relu2.forward())
