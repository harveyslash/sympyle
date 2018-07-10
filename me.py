from Nodes import Tensor, Matmul, MSE, Relu, Add, SoftmaxWithCrossEntropy
from Nodes import Node
import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(100)

X = np.linspace(10, 10, 100).reshape(-1, 2).astype(np.float32)

Y_temp = np.random.randint(0, 2, size=X.shape[:1])
Y = np.zeros(shape=(50, 2))

Y[np.arange(50), Y_temp] = 1

X_t = Tensor(X)
Y_t = Tensor(Y)

output = SoftmaxWithCrossEntropy(X_t, Y_t)
# print(I_t.value)
# print(I_t2.value)

# branch1 = head1 @ Tensor(I)
# branch2 = head1 @ Tensor(I)
# loss1 = MSE(branch1, T_t)
# loss2 = MSE(branch2, T_t2)
# final = Tensor(I) + loss2  # + loss2 + loss1 + loss1 + loss1 + loss1
# final = Add(Tensor(I), loss2)
# final = loss1 + loss2 + loss2


for i in range(10):
    output_val = output.forward()
    print(output_val)

    output.backward()
    # print(X_t.backward_val)
    grad = X_t.backward_val
    # print(grad)

    X_t.value -= 1 * grad
    output.clear()
# print(X_t.value)
# print(X_t.backward_val * X_t.value)
# output.backward(parent_grads=np.array([1.0]))
# print(I_t.backward_val)
output.draw_graph("gaha.png")

# print(Node.consts.no_grads)
