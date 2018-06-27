from Nodes import Tensor, Matmul, MSE, Relu, Add
import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(100)

X = np.linspace(-np.pi, np.pi, 1).reshape(-1, 1).astype(np.float32)
X_copy = X.copy()
pure = np.cos(X)

Y = pure + np.random.randn(*X.shape) * .07
Y = Y.astype(np.float32)

I = X.reshape(-1, 1)  # an input with minibatch size 100 and 1 feature

L1 = np.random.randn(1, 1).astype(np.float32)
L1_b = np.random.randn(1, 1).astype(np.float32)
L1_b2 = np.random.randn(1, 1).astype(np.float32)

T = Y.reshape(-1, 1)

I_t = Tensor(I)
I_t2 = Tensor(I)
T_t = Tensor(T)
T_t2 = Tensor(T)

head1 = Tensor(I) + Tensor(I)

branch1 = head1 @ Tensor(I)
branch2 = head1 @ Tensor(I)
loss1 = MSE(branch1, T_t)
loss2 = MSE(branch2, T_t2)
final = loss1 + loss2 + loss2 + loss1
# final = loss1 + loss2 + loss2

# output.backward()
# print(I_t.backward_val)
final.draw_graph("gaha.png")
