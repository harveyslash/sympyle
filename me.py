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

T = Y.reshape(-1, 1)

I_t = Tensor(I)
T_t = Tensor(T)

L1_t = Tensor(L1)
L1_bt = Tensor(L1_b)
# L1_bt1 = Tensor(L1_b)
# L1_bt2 = Tensor(L1_b)
# L1_bt3 = Tensor(L1_b)
# L1_bt4 = Tensor(L1_b)
# L1_bt5 = Tensor(L1_b)
# L1_bt6 = Tensor(L1_b)

# matmul1 = I_t @ L1_t
added = L1_bt - L1_t
# added2 = L1_bt + L1_t
# added2 = L1_bt1 + L1_t
# added3 = L1_bt1 + L1_t
# final = added  # + added2

# mse = MSE(added, T_t)
# mse2 = MSE(added, T_t)
# # mse2 = MSE(added, T_t)
# # mse2 = MSE(added, T_t)
# # mse2 = MSE(added, T_t)
# # mse2 = MSE(added, T_t)
# # final = added + added2 #+ added3
# # mse2 = MSE(added, T_t)
# #
# final = mse + mse2  # + mse2 + mse2 + mse2
# ALA = MSE(final, T_t)
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv

# G = pgv.AGraph(directed=True)
# G.layout('dot')

# mse.backward()
# mse2.backward()
# mse.backward()
# mse.backward()
# mse.backward()
# mse.backward()
# print(mse.forward())
# print(L1_bt.backward_val)
# print(L1_bt.backward_val)
# print(final.forward())
# mse.backward()
# final.backward()
# print(L1_t.backward_val)
# print(L1_bt.backward_val)
added.backward()
# print("AAAA")
print(L1_t.backward_val)
print(L1_bt.backward_val)
# print(L1_bt1.backward_val)
added.draw_graph("gaha.png")

# print(mse.children)
# print(added.children)
# G.draw('file.png', format='png', prog='dot')
# G.write('graph.dot')
# G.draw(path='graphy.png', format='png')

# # same layout using matplotlib with no labels
# plt.title('draw_networkx')
# pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=False, arrows=False)
# plt.savefig('nx_test.png')
