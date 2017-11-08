import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
print(t)

print(t.ndim)
print(t.shape)

print(t[0], t[1], t[-1])

print(t[2:5], t[4:-1])

print(t[2:],t[:3])

t2 = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7.]])

print(t2)

print(t2.ndim)
print(t2.shape)
