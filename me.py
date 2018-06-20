import numpy as np

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 0]]).T
X = np.random.rand(100, 3)
y = np.random.rand(100, 1)
print(y.shape)

syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1
for j in range(60000):
    l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))
    print(l1.shape)
    print("INPUT SHAPE")
    print(X.shape)

    print("WEIGHT SHAPE")
    print(syn0.shape)

    print("calling after dot shape")
    print(np.matmul(X, syn0).shape)
    print("-"*80)
    l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))
    print(l2.shape)
    l2_delta = (y - l2) * (l2 * (1 - l2))
    print(l2_delta.shape)
    # exit()
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))
    print(l1_delta.shape)
    print(l1.T.dot(l2_delta).shape)
    print(l2_delta.shape)
    exit()
    # syn1 += l1.T.dot(l2_delta)
    # syn0 += X.T.dot(l1_delta)
    # print(y - l2)
    exit()
