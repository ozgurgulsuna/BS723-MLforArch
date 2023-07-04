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

import datetime
import os
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
save = False

filename = "d:/2022-23/Okul/Dersler/BS723/[05][03] Experimentation/makingEvaluation/chaotic_attractors.json"
# filename = "chaotic_attractors.json"

with open(filename, 'r') as read_file:
    data = json.load(read_file)

scale_A = data[system_A]["scale"][:3]
offset_A = data[system_A]["offset"][:3]

scale_B = data[system_B]["scale"][:3]
offset_B = data[system_B]["offset"][:3]

model_A = eval(system_A+"()")
model_B = eval(system_B+"()")


result_dir = "d:/2022-23/Okul/Dersler/BS723/[05][03] Experimentation/makingEvaluation/results/"
if os.path.isdir(result_dir):
    pass
else:
    os.mkdir(result_dir)

logs = os.path.join(result_dir, "logs.txt")

if os.path.isfile(logs):
    pass
else:
    with open(logs, 'w') as f:
        f.write("")
        f.close()


# Initial conditions ==========================================================================================================#

range = 0.1
x0, y0, z0 = (np.random.rand()*range-range/2,np.random.rand()*range-range/2,np.random.rand()*range-range/2)
x1, y1, z1 = (np.random.rand()*range-range/2,np.random.rand()*range-range/2,np.random.rand()*range-range/2)
x2, y2, z2 = (np.random.rand()*range-range/2,np.random.rand()*range-range/2,np.random.rand()*range-range/2)

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

# a = -3.714534137284764 
# b =14.632861553314914 
# c = 6.156366208046949 
# d = 0.01961998110523311

# a = -17.674702003127702 
# b = 21.764906670550904
# c = -24.344549422173795
# d = 0.04872006514863636

a = 11.677448009175512 
b = 10.101709495239735 
c = 0.021031560848727704 
d = -0.24451814258624305

# Falloff function ============================================================================================================#

def falloff(distance):
    return 0.5/(1+(distance*10)**2)

def distance(plane, point):
    a,b,c,d = plane
    x,y,z = point
    return abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)



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

# Solve the systems ===========================================================================================================#

Merge4 = odeint(merge, (x0, y0, z0), t)
Merge5 = odeint(merge, (x1, y1, z1), t)
Merge6 = odeint(merge, (x2, y2, z2), t)


# Save & Print The Tag ========================================================================================================#
date_created = datetime.datetime.now()
uid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

f = open(logs, "a")
tag = "================================================================================================================="+"\n"
tag += "model_A: "+str(model_A)+"\n"
tag += "model_B: "+str(model_B)+"\n"
tag += "plane_parameters: a = "+str(a)+" b = "+str(b)+" c = "+str(c)+" d = "+str(d)+"\n"
tag += "other_parameters: t = "+str(dt*len(t))+", dt = "+str(dt)+", smooth = "+str(smooth)+", saved ="+str(save) +"\n"
tag += "initial_conditions: x0= "+str(x0)+" y0= "+str(y0)+" z0= "+str(z0)+"\n"
tag += "initial_conditions: x1= "+str(x1)+" y1= "+str(y1)+" z1= "+str(z1)+"\n"
tag += "initial_conditions: x2= "+str(x2)+" y2= "+str(y2)+" z2= "+str(z2)+"\n"
tag += "date_created: "+str(date_created)+"\n"
tag += "uid: "+str(uid)+"\n"
tag += "notes: "+str("None")+"\n"
tag += "================================================================================================================="+"\n"
print(tag)
f.write(tag)
f.close()

# Save the data ===============================================================================================================#
if save:
    print("Saving the data...")
    np.savetxt(os.path.join(result_dir, str(uid)+"_Merge4.csv"), Merge4, delimiter=",")
    np.savetxt(os.path.join(result_dir, str(uid)+"_Merge5.csv"), Merge5, delimiter=",")
    np.savetxt(os.path.join(result_dir, str(uid)+"_Merge6.csv"), Merge6, delimiter=",")
else:
    pass

# Plot the system =============================================================================================================#

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


if save:
    plt.savefig(os.path.join(result_dir, str(uid)+".png"))
else:
    pass

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




