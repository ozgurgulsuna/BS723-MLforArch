#==============================================================================================================================#
#                                                                                                                              #
#                                            C O N T I N U O U S   M E R G I N G                                               #
#                                            S T R A N G E   A T T R A C T O R S                                               #   
#                                                                                                                              #                                 
#       Author: ozgurgulsuna                                                                                                   #                                                                            |
#       Date: 02.07.2023                                                                                                       #                            
#                                                                                                                              # 
#       Description:                                                                                                           #
#         This code implements a continuous merging technique for strange attractors, inspired by the mesmerizing              #
#         patterns found in chaotic systems. The algorithm blends multiple attractors together, resulting in intricate         #
#         visual representations. This project aims to explore the beauty of chaos and the complex dynamics of                 #
#         nonlinear systems.                                                                                                   #
#                                                                                                                              #   
#==============================================================================================================================#

# Imports =====================================================================================================================#

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

import json 
from flows import *

# Transformations =============================================================================================================#

def Normalize(state, scale, offset):
    # Normalize the point to the unit cube
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
    # Expand the point to the original scale
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

# Parameters ==================================================================================================================#

dt = 0.001
t = np.arange(0, dt*100000, dt)

# Read Data ===================================================================================================================#

system_A = "Halvorsen"
system_B = "GenesioTesi"

single = 0
smooth = True
portal_plane = False

filename = "d:/2022-23/Okul/Dersler/BS723/[05][03] Experimentation/moreSystems/chaotic_attractors.json"
# filename = "chaotic_attractors.json"

with open(filename, 'r') as read_file:
    data = json.load(read_file)

scale_A = data[system_A]["scale"][:3]
offset_A = data[system_A]["offset"][:3]

scale_B = data[system_B]["scale"][:3]
offset_B = data[system_B]["offset"][:3]

model_A = eval(system_A+"()")
model_B = eval(system_B+"()")





# Initial conditions ==========================================================================================================#
x0,y0,z0 = Normalize((0.1,0.1,0.1),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
x1,y1,z1 = Normalize((np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
x2,y2,z2 = Normalize((np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5),(136.6145132443005,132.93521309671516,83.11710323466052), (0.016690981504353886,-0.054503652638672406,33.21631003781814))
x0,y0,z0 = Normalize((np.random.rand()-0.5,np.random.rand()-0.5,np.random.rand()-0.5),(136.6145132443005,132.93521309671516,83.11710323466052),  (0.016690981504353886,-0.054503652638672406,33.21631003781814))

# x0,y0,z0 = Normalize((-0.21422356,-0.17935495,14.105516),scale_B, offset_B)


a=(np.random.rand()-0.5)*50
b=(np.random.rand()-0.5)*50
c=(np.random.rand()-0.5)*50
d=np.random.rand()-0.5

# a = -1.515980368201647
# b= -21.35202216446338
# c = -5.822262341841345
# d = 0.36183045383970425

# a = -17.789229828346574
# b = -24.07603117506562
# c = 14.345434996525968
# d = 0.34105428120779835

a = -3.714534137284764 
b =14.632861553314914 
c = 6.156366208046949 
d = 0.01961998110523311


#----------------------------------------------------------#
# Falloff function
def falloff(distance):
    return 0.5/(1+(distance*10)**2)

def distance(plane, point):
    a,b,c,d = plane
    x,y,z = point
    return abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)


# Define the systems
# def TSUCS1(state, t):
#     x, y, z = state
#     return 40 * (y - x) + 0.5 * x * z, 20 * y - x * z , 0.833 * z + x * y - 0.65 * x**2


# def YuWang(state, t):
#     x, y, z = state
#     return 10*(y-x), 40*x-2*x*z, np.exp(x*y)-2.5*z
  

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

def single_system(state,t):
    x_0,y_0,z_0 = state

    x,y,z = Expand(state, scale_A, offset_A)

    dx,dy,dz = model_A.rhs(np.asarray([x,y,z]),0)

    x = dx*dt +x
    y = dy*dt +y
    z = dz*dt +z

    x,y,z = Normalize((x,y,z), scale_A, offset_A)
                 
    dx = (x-x_0)/dt
    dy = (y-y_0)/dt
    dz = (z-z_0)/dt

    return dx,dy,dz  

def merge(state,t):
    x_0,y_0,z_0 = state 

    dist = distance((a,b,c,d),(x_0,y_0,z_0))
    fall = falloff(dist)

    x,y,z = Expand(state, scale_A, offset_A)

    dx,dy,dz = model_A.rhs(np.asarray([x,y,z]),0)[:3]

    x = dx*dt +x
    y = dy*dt +y
    z = dz*dt +z

    x,y,z = Normalize((x,y,z), scale_A, offset_A)
                 
    dx_a = (x-x_0)/dt
    dy_a = (y-y_0)/dt
    dz_a = (z-z_0)/dt

    x,y,z = Expand(state, scale_B, offset_B)

    dx,dy,dz = model_B.rhs(np.asarray([x,y,z]),0)[:3]

    x = dx*dt +x
    y = dy*dt +y
    z = dz*dt +z

    x,y,z = Normalize((x,y,z), scale_B, offset_B)
                 
    dx_b = (x-x_0)/dt
    dy_b = (y-y_0)/dt
    dz_b = (z-z_0)/dt

    if (single!= 0):
        if single == "A":
            return dx_a, dy_a, dz_a
        elif single == "B":
            return dx_b, dy_b, dz_b

    elif (a*x+b*y+c*z+d < 0):

        if smooth:
            dx=dx_a*fall+dx_b*(1-fall)
            dy=dy_a*fall+dy_b*(1-fall)
            dz=dz_a*fall+dz_b*(1-fall)
        else:
            dx=dx_a
            dy=dy_a
            dz=dz_a

        return dx,dy,dz
    

    else:
        if smooth:
            dx=dx_a*(1-fall)+dx_b*fall
            dy=dy_a*(1-fall)+dy_b*fall
            dz=dz_a*(1-fall)+dz_b*fall
        else:
            dx=dx_b
            dy=dy_b
            dz=dz_b

        return dx,dy,dz


#----------------------------------------------------------#
# Solve the systems

# Merge1 = odeint(Merge, (x0, y0, z0), t)

# Merge2 = odeint(Merge, (x1, y1, z1), t)

# Merge3 = odeint(Merge, (x2, y2, z2), t)

# singleass = odeint(single_system, (x0, y0, z0), t)

Merge4 = odeint(merge, (x0, y0, z0), t)
Merge5 = odeint(merge, (x1, y1, z1), t)
Merge6 = odeint(merge, (x2, y2, z2), t)

# Merge2 = odeint(Merge, (x2, y2, z2), t)


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


# # Plot the system
# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(Merge1[:, 0], Merge1[:, 1], Merge1[:, 2])
# ax.plot3D(Merge3[:, 0], Merge3[:, 1], Merge3[:, 2], color="green")
# ax.plot3D(Merge2[:, 0], Merge2[:, 1], Merge2[:, 2], color="red")
# # ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()



# # Plot the system
# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(singleass[:, 0], singleass[:, 1], singleass[:, 2])
# # ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

# Plot the system
print("Plotting the system...")
plt.figure()
ax = plt.axes(projection='3d')

if portal_plane:
    x = np.linspace(-0.5, 0.5, 2)
    y = np.linspace(-0.5, 0.5, 2)
    x, y = np.meshgrid(x, y)
    eq = -a*x/c - b*y/c  - d/c
    ax.plot_surface(x, y, eq, alpha=0.2)



ax.plot3D(Merge4[:, 0], Merge4[:, 1], Merge4[:, 2], color="black")
ax.plot3D(Merge5[:, 0], Merge5[:, 1], Merge5[:, 2], color="lightslategray")
ax.plot3D(Merge6[:, 0], Merge6[:, 1], Merge6[:, 2], color="darkslategray")

# ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


#
#       _____                _____            __  __  __
#      /\    \              /\    \          /\ \/ / /\ \
#     /::\    \            /::\____\        /::\  / /  \ \
#    /::::\    \          /:::/    /       /:/\:\/ /    \ \
#   /::::::\    \        /:::/    /       /:/  \:\_\____\ \
#  /:::/\:::\    \      /:::/    /       /:/    \:\/___/ \
# /:::/  \:::\    \    /:::/    /       /:/     \:\__\   _
# \::/    \:::\    \  /:::/    /       /:/      \/__/  /\ \
#  \/____/ \:::\    \/:::/    /        \/_____/      /::\ \
#           \:::\____\/:::/    /                      /:/\:\ \
#            \::/    /\::/    /                      /:/__\:\ \
#             \/____/  \/____/                       \:\   \:\ \
#                                                      \:\   \:\ \
#                                                       \:\   \:\_\
#                                                        \:\__\::/
#                                                         \/__\:/ 




