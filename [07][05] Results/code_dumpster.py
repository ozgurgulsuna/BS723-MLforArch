










def rotate_points_in3D(points,theta_x,theta_y,theta_z):
    theta_x = np.radians(theta_x)
    theta_y = np.radians(theta_y)
    theta_z = np.radians(theta_z)
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
    rotated_points = np.dot(R, points.T).T
    return rotated_points








a = 0.5
b = 0.5
c = 0.01
# VecStart_x = [a,-a,-a,a]
# VecStart_y = [b,b,-b,-b]
# VecStart_z = [c,c,c,c]

# VecEnd_x = [-a,-a,a,a]
# VecEnd_y = [b,-b,-b,b]
# VecEnd_z = [c,c,c,c,]   

# for i in range(4):
#     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],[VecStart_z[i],VecEnd_z[i]], color="firebrick", linewidth=2,zorder=9999,alpha=0.5)



points = [[a,b,c],[-a,b,c],[-a,-b,c]]

points = rotate_points_in3D(np.array(points),np.pi/2,np.pi/2,np.pi/2)


p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)

d = -point.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt.figure()
ax =plt.axes(projection='3d')
ax.plot_surface(xx, yy, z)
plt.show()
