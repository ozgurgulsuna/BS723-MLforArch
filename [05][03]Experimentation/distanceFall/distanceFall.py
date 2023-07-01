

# to be installed

# pip install scipy
# pip install matplotlib
# pip install numpy

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




# solver = 'euler'
dt = 0.001
t = np.arange(0, dt*10000, dt)

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



# sp1=-13.383584125755775
# sp2=7.753965912068551
# off3 = 0.2936950556786103

  

# Parameters
# a = 40
# c = 0.833
# d = 0.5
# e = 0.65
# f = 20

# Yu-Wang Attractor
# Initial values
# x1 = 0.1
# y1 = 0.1
# z1 = 0.1

# Parameters
a1 = 10
b1 = 40
c1 = 2
d1 = 2.5

#----------------------------------------------------------#
# Falloff function
def falloff(distance):
    return 1/(1+distance**2)

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
    return a1*(y-x), b1*x-c1*x*z, np.exp(x*y)-d1*z

# def Merge(state,t):
#     x, y, z = state
#     if (t < 10):
#         return a * (y - x) + d * x * z, f * y - x * z , c * z + x * y - e * x**2
#     else:
#         return a1*(y-x), b1*x-c1*x*z, np.exp(x*y)-d1*z
    

def myTSUCS1(state,t):
    # x,y,z = state
    x,y,z = Expand(state, (136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))

    dx = (40 * (y - x) + 0.5 * x * z)
    dy = (20 * y - x * z)
    dz = (0.833 * z + x * y - 0.65 * x**2)

    x = dx*dt+x
    y = dy*dt+y
    z = dz*dt+z

    x,y,z = Normalize((x,y,z),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
    return x,y,z

print(a,b,c,d)
def Merge(state,t):
    x,y,z = state 
   
    if (a*x+b*y+c*z+d < 0):
        x,y,z = Expand(state, (136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))

        dx = (40 * (y - x) + 0.5 * x * z)
        dy = (20 * y - x * z)
        dz = (0.833 * z + x * y - 0.65 * x**2)

        x = dx*dt+x
        y = dy*dt+y
        z = dz*dt+z

        x,y,z = Normalize((x,y,z),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
        return x,y,z
    else:
        x,y,z = Expand(state, (4.399089600367304,8.252102382380636,57.14753811199316), (0.016690981504353886,0.21957693624264607,28.57376905599658))
        
        dx = a1*(y-x)
        dy = b1*x-c1*x*z
        dz = np.exp(x*y)-d1*z

        x = dx*dt+x
        y = dy*dt+y
        z = dz*dt+z

        x,y,z = Normalize((x,y,z), (4.399089600367304,8.252102382380636,57.14753811199316), (0.016690981504353886,0.21957693624264607,28.57376905599658))
        return x,y,z       



# def Merge(state,t):
#     x, y, z = state
#     if (x > 0) :
#         return a * (y - x) + d * x * z, f * y - x * z , c * z + x * y - e * x**2
#     else:
#         return a*1.04 * (y - x) + d*1.04 * x * z, f*1.04 * y - x * z , c * z + x * y - e*1.04 * x**2

#----------------------------------------------------------#
# mysolver
def mysolver(system, state, t):
    x, y, z = state
    output = np.empty((1,3))
    for i in range(len(t)):
        x, y, z = system((x, y, z), t[i])
        output = np.append(output, [[x,y,z]], axis=0)
        # print(x,y,z)

    return output


myTSUCS1 = mysolver(myTSUCS1, (x0, y0, z0), t)
myTSUCS1 = np.delete(myTSUCS1, 0, 0)

myMerger = mysolver(Merge, (x0, y0, z0), t)
myMerger = np.delete(myMerger, 0, 0)

myMerger1 = mysolver(Merge, (x1, y1, z1), t)
myMerger1 = np.delete(myMerger1, 0, 0)

myMerger2 = mysolver(Merge, (x2, y2, z2), t)
myMerger2 = np.delete(myMerger2, 0, 0)



#----------------------------------------------------------#
# Solve the systems
TSUCS1 = odeint(TSUCS1, (x0, y0, z0), t)

YuWang = odeint(YuWang, (x1, y1, z1), t)

Merge = odeint(Merge, (x0, y0, z0), t)
# np.delete(Merge[:,0],-1)
# np.delete(Merge[:,1],-1)
# np.delete(Merge[:,2],-1)
#----------------------------------------------------------#

# Plot the system

# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(TSUCS1[:, 0], TSUCS1[:, 1], TSUCS1[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

#----------------------------------------------------------#

# Yu-Wang Attractor

plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(YuWang[:, 0], YuWang[:, 1], YuWang[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

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

ax.plot3D(myMerger[:, 0], myMerger[:, 1], myMerger[:, 2])
ax.plot3D(myMerger1[:, 0], myMerger1[:, 1], myMerger1[:, 2], color="green")
ax.plot3D(myMerger2[:, 0], myMerger2[:, 1], myMerger2[:, 2], color="red")
# ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#----------------------------------------------------------#




