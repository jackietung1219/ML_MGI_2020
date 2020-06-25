#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
data = np.loadtxt('XYZ.csv', delimiter=',')
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
plt.xlim((-100, 100))
plt.ylim((-100, 100))
ax.set_zlim(-5, 45)
plt.show()

# %%
