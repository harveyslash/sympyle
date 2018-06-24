from Nodes import Tensor, Matmul, MSE, Relu, Add
import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(100)

X = np.linspace(-np.pi, np.pi, 800).reshape(-1, 1).astype(np.float32)
X_copy = X.copy()
pure = np.cos(X)

Y = pure + np.random.randn(*X.shape) * .07
Y = Y.astype(np.float32)

I = X.reshape(-1, 1)  # an input with minibatch size 100 and 1 feature

L1 = np.random.randn(1, 1).astype(np.float32)
L1_b = np.random.randn(1, 100).astype(np.float32)

T = Y.reshape(-1, 1)

I_t = Tensor(I)
T_t = Tensor(T)

L1_t = Tensor(L1)
L1_bt = Tensor(L1_b)

matmul1 = I_t @ L1_t

mse = MSE(matmul1, T_t)

import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv

# G = pgv.AGraph(directed=True)
# G.layout('dot')

mse.construct_graph("gaha.png")

# G.draw('file.png', format='png', prog='dot')
# G.write('graph.dot')
# G.draw(path='graphy.png', format='png')

# # same layout using matplotlib with no labels
# plt.title('draw_networkx')
# pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=False, arrows=False)
# plt.savefig('nx_test.png')
