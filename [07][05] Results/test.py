import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

X = np.array([1, 1, -1, -1])
Y = np.array([1, -1, -1, 1])
Z = np.array([0, 0, 0, 0])

def rotate_points( X, Y, Z, theta_x, theta_y, theta_z ):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    rotated_points = np.dot(R, np.array([X,Y,Z]))
    return rotated_points

X, Y, Z = rotate_points( X, Y, Z, 0, np.pi/1.999, 0 )




fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
ax.set_zlim(-1, 1)
surf1 = ax.plot_trisurf(X, Y, Z, antialiased=True, shade=False, alpha=0.5)

plt.show()