# test
import numpy as np
import matplotlib.pyplot as plt

r_v = np.ones((4,2))
print(r_v)
print(r_v[0:3,0] - r_v[3,0])


# a = 20
# b = 10
# f = np.linspace(0, 2*np.pi, 1000)

# e = np.sqrt(1-(b/a)**2)

# r = a*(1-e**2)/(1+e*np.cos(f))


# x = np.linspace(-1.0, 1.0, 1000)
# y = np.linspace(-1.0, 1.0, 1000)

# x = r*np.cos(f)
# y = r*np.sin(f)

# # X, Y = np.meshgrid(x,y)
# # F = X**2 + Y**2 - 0.6
# plt.plot(x,y)
# # plt.contour(X,Y,[0])
# plt.axis('equal')
# plt.show()


# plt.plot(f, func)
# plt.show()

# a = np.arange(20).reshape(4,5)
# print(a)

# # b = np.zeros((4,5))
# # print(b)

# c = np.ones((4,5))

# d = np.where(a<10)
# print(d)

# d = np.where((5<a) & (a<10))
# print(d)
