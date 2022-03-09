import numpy as np
a = np.array([3, 4, 1, 2, 8, 2, 4, 9])
b = np.array([[3, 4, 1, 2], [8, 2, 4, 9]])
c = np.zeros((3, 5))
d = np.ones((3, 5, 2))
e = np.random.rand(3, 5)
f = np.array([[3, 4, 1, 2], [8, 2, 4, 9], [2, 1, 8, 7]])
g = np.array([[True, False, True, False],
              [True, False, True, False],
              [True, False, True, False]])
print(f[f<5])
print(f<5)
print(f.shape)

print(c.shape)
c = np.zeros(3)
c = np.expand_dims(n, 1)
print(c)
print(c.shape)
c = np.expand_dims(n, -1)
print(c.shape)

print(a)
h = np.reshape(a, (2, 4))
print(h)
