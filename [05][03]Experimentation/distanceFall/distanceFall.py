import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

#----------------------------------------------------------#
# 1. TSUCS 1 Attractor (Three-Scroll Unified Chaotic System)
# 2. Yu-Wang Attractor
#----------------------------------------------------------#

#----------------------------------------------------------#
# Define Transformations

def Normalize(state, scale, offset):
    x, y, z = state
    scale_x, scale_y, scale_z = scale
    offset_x, offset_y, offset_z = offset

    x = x - offset_x
    y = y - offset_y
    z = z - offset_z

    x = x/scale_x
    y = y/scale_y
    z = z/scale_z

    return x,y,z

def Expand(state, scale, offset):
    x, y, z = state
    scale_x, scale_y, scale_z = scale
    offset_x, offset_y, offset_z = offset

    x = x * scale_x
    y = y * scale_y
    z = z * scale_z

    x = x + offset_x
    y = y + offset_y
    z = z + offset_z

    return x,y,z

dt = 0.0005
t = np.arange(0, dt*75000, dt)

#----------------------------------------------------------#
# TSUCS 1 Attractor (Three-Scroll Unified Chaotic System)
# Initial values


x0,y0,z0 = Normalize((0.1,0.1,0.1),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
x1,y1,z1 = Normalize((np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
x2,y2,z2 = Normalize((np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
x0,y0,z0 = Normalize((np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5),(136.6145132443005,132.93521309671516,83.11710323466052),  (0.016690981504353886,-0.054503652638672406,33.21631003781814))


a=(np.random.rand()-0.5)*50
b=(np.random.rand()-0.5)*50
c=(np.random.rand()-0.5)*50
d=np.random.rand()-0.5


#----------------------------------------------------------#
# Falloff function
def falloff(distance):
    return 0.5/(1+(distance*10)**2)

def distance(plane, point):
    a,b,c,d = plane
    x,y,z = point
    return abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)


# Define the systems
def TSUCS1(state, t):
    x, y, z = state
    return 40 * (y - x) + 0.5 * x * z, 20 * y - x * z , 0.833 * z + x * y - 0.65 * x**2


def YuWang(state, t):
    x, y, z = state
    return 10*(y-x), 40*x-2*x*z, np.exp(x*y)-2.5*z
  

# def myTSUCS1(state,t):
#     x_0,y_0,z_0 = state
#     x,y,z = Expand(state, (136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))

#     x = (40 * (y - x) + 0.5 * x * z)*dt +x
#     y = (20 * y - x * z)*dt +y
#     z = (0.833 * z + x * y - 0.65 * x**2)*dt +z

#     x,y,z = Normalize((x,y,z),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))

#     dx = (x-x_0)/dt
#     dy = (y-y_0)/dt
#     dz = (z-z_0)/dt

#     return dx,dy,dz

print(a,b,c,d)
def Merge(state,t):
    x_0,y_0,z_0 = state 

    dist = distance((a,b,c,d),(x_0,y_0,z_0))
    fall = falloff(dist)


    x,y,z = Expand(state, (136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))

    x = (40 * (y - x) + 0.5 * x * z)*dt +x
    y = (20 * y - x * z)*dt +y
    z = (0.833 * z + x * y - 0.65 * x**2)*dt +z
    #  print(dx,dy,dz)
    
    x,y,z = Normalize((x,y,z),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))

    dx_a = (x-x_0)/dt
    dy_a = (y-y_0)/dt
    dz_a = (z-z_0)/dt

    x,y,z = Expand(state, (4.399089600367304,8.252102382380636,57.14753811199316), (0.016690981504353886,0.21957693624264607,28.57376905599658))

    x = (10*(y-x))*dt +x
    y = (40*x-2*x*z)*dt +y
    z = (np.exp(x*y)-2.5*z)*dt +z
    #  print(dx,dy,dz)

    x,y,z = Normalize((x,y,z), (4.399089600367304,8.252102382380636,57.14753811199316), (0.016690981504353886,0.21957693624264607,28.57376905599658))

    dx_b = (x-x_0)/dt
    dy_b = (y-y_0)/dt
    dz_b = (z-z_0)/dt

    if (a*x+b*y+c*z+d < 0):

        dx=dx_a*fall+dx_b*(1-fall)
        dy=dy_a*fall+dy_b*(1-fall)
        dz=dz_a*fall+dz_b*(1-fall)

        return dx,dy,dz

    else:

        dx=dx_a*(1-fall)+dx_b*fall
        dy=dy_a*(1-fall)+dy_b*fall
        dz=dz_a*(1-fall)+dz_b*fall

        return dx,dy,dz     

#----------------------------------------------------------#
# Solve the systems
TSUCS1 = odeint(TSUCS1, (x0, y0, z0), t)

YuWang = odeint(YuWang, (x1, y1, z1), t)

Merge1 = odeint(Merge, (x0, y0, z0), t)

Merge2 = odeint(Merge, (x1, y1, z1), t)

Merge3 = odeint(Merge, (x2, y2, z2), t)

# Merge2 = odeint(Merge, (x2, y2, z2), t)


#----------------------------------------------------------#

# Plot the system

print("Plotting the system...")
plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(TSUCS1[:, 0], TSUCS1[:, 1], TSUCS1[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#----------------------------------------------------------#

# Yu-Wang Attractor

# plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(YuWang[:, 0], YuWang[:, 1], YuWang[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

#----------------------------------------------------------#


# Plot the system
# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(Merge[:, 0], Merge[:, 1], Merge[:, 2])
# # ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

#----------------------------------------------------------#

# Plot the system
# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(myTSUCS1[:, 0], myTSUCS1[:, 1], myTSUCS1[:, 2])
# # ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

#----------------------------------------------------------#


# Plot the system
print("Plotting the system...")
plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(Merge1[:, 0], Merge1[:, 1], Merge1[:, 2])
ax.plot3D(Merge3[:, 0], Merge3[:, 1], Merge3[:, 2], color="green")
ax.plot3D(Merge2[:, 0], Merge2[:, 1], Merge2[:, 2], color="red")
# ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#----------------------------------------------------------#




