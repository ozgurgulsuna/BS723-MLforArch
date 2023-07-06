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


# Parameters ==================================================================================================================#

dt = 0.01
t = np.arange(0, dt*800*2, dt)

system_A = "Halvorsen"
system_B = "GuckenheimerHolmes"

num_ic = 10

single = "B"
smooth = True
portal_plane = False
save = True

color_set = "red"
# color_set = "red"


# Directory ===================================================================================================================#

filename = "d:/2022-23/Okul/Dersler/BS723/[07][05] Results/chaotic_attractors.json"
result_dir = "d:/2022-23/Okul/Dersler/BS723/[07][05] Results/results/"

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


# Read data from json file ====================================================================================================#

with open(filename, 'r') as read_file:
    data = json.load(read_file)

scale_A = data[system_A]["scale"][:3]
offset_A = data[system_A]["offset"][:3]

scale_B = data[system_B]["scale"][:3]
offset_B = data[system_B]["offset"][:3]

model_A = eval(system_A+"()")
model_B = eval(system_B+"()")


# Initial conditions ==========================================================================================================#

rang = 0.1
x0, y0, z0 = (np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2)
x1, y1, z1 = (np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2)
x2, y2, z2 = (np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2)

a = 0
b = 0
c = 1
d = 0
# d = np.random.rand()*0.25-0.25

r = np.random.rand()*np.pi*2

theta = np.random.rand()*np.pi*2

phi = np.random.rand()*np.pi*2


# r = 0.7108928822911016

# theta = 0.2701111537255759

# phi = 0.21166722761979148
# Random rotation =============================================================================================================#

from scipy.spatial.transform import Rotation


rotation_r = Rotation.from_rotvec((r)* np.array([1, 0.01, 0.01]))
rotation_theta = Rotation.from_rotvec((theta)* np.array([0.01, 1, 0.01]))
rotation_phi = Rotation.from_rotvec((phi)* np.array([0.01, 0.01, 1]))

# Define a normal vector for the plane
normal_vector = np.array([a, b, c])

# Normalize the normal vector
normal_vector = normal_vector / np.linalg.norm(normal_vector)

# Rotate the normal vector
rotated_normal_vector = rotation_r.apply(normal_vector)
rotated_normal_vector = rotation_theta.apply(rotated_normal_vector)
rotated_normal_vector = rotation_phi.apply(rotated_normal_vector)

a,b,c = rotated_normal_vector[0],rotated_normal_vector[1],rotated_normal_vector[2]


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


# Falloff function ============================================================================================================#

def falloff(distance):
    return 0.5/(1+(distance*10)**2)

def distance(plane, point):
    a,b,c,d = plane
    x,y,z = point
    return abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)


# Merge function ==============================================================================================================#
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

outputs = []
for i in range(num_ic):
    x0, y0, z0 = (np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2,np.random.rand()*rang-rang/2)
    outputs.append(odeint(merge, (x0, y0, z0), t))


# Merge4 = odeint(merge, (x0, y0, z0), t)
# Merge5 = odeint(merge, (x1, y1, z1), t)
# Merge6 = odeint(merge, (x2, y2, z2), t)


# Evaluate the system =========================================================================================================#

from scipy.stats import linregress
def calculate_lyapunov_exponent(traj1, traj2, dt=1.0):
    """
    Calculate the lyapunov exponent of two multidimensional trajectories using
    simple linear regression based on the log-transformed separation of the
    trajectories.

    Args:
        traj1 (np.ndarray): trajectory 1 with shape (n_timesteps, n_dimensions)
        traj2 (np.ndarray): trajectory 2 with shape (n_timesteps, n_dimensions)
        dt (float): time step between timesteps

    Returns:
        float: lyapunov exponent
    """
    separation = np.linalg.norm(traj1 - traj2, axis=1)
    log_separation = np.log(separation)
    time_vals = np.arange(log_separation.shape[0])
    slope, intercept, r_value, p_value, std_err = linregress(time_vals, log_separation)
    lyap = slope / dt
    return lyap

def lyap_exp(x, y, z, dt, n):
    lyap = []
    for i in range(n):
        print("Iteration: ", i)
        Merge4 = odeint(merge, (x, y, z), t)
        Merge5 = odeint(merge, (x+0.0001, y, z), t)
        lyap.append(calculate_lyapunov_exponent(Merge4, Merge5, dt))
        x = Merge4[-1, 0]
        y = Merge4[-1, 1]
        z = Merge4[-1, 2]
        print("Lyapunov exponent: ", lyap[-1])
    return np.mean(lyap)

lyap = lyap_exp(x0, y0, z0, dt, 10)


# Save & Print The Tag ========================================================================================================#

date_created = datetime.datetime.now()
uid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

f = open(logs, "a")
tag = "================================================================================================================="+"\n"
tag += "model_A: "+str(model_A)+"\n"
tag += "model_B: "+str(model_B)+"\n"
tag += "plane_parameters: a = "+str(a)+" b = "+str(b)+" c = "+str(c)+" d = "+str(d)+"\n"
tag += "other_parameters: t = "+str(dt*len(t))+", dt = "+str(dt)+", smooth = "+str(smooth)+", saved = "+str(save) +"\n"
tag += "initial_conditions: x0= "+str(x0)+" y0= "+str(y0)+" z0= "+str(z0)+"\n"
tag += "initial_conditions: x1= "+str(x1)+" y1= "+str(y1)+" z1= "+str(z1)+"\n"
tag += "initial_conditions: x2= "+str(x2)+" y2= "+str(y2)+" z2= "+str(z2)+"\n"
tag += "date_created: "+str(date_created)+"\n"
tag += "uid: "+str(uid)+"\n"
tag += "lyapunov_exponent: "+str(lyap)+"\n"
tag += "================================================================================================================="+"\n"
print(tag)
f.write(tag)
f.close()


# Save the data ===============================================================================================================#
record = []
if save:
    for i in range(num_ic):
        # combine timeseries together with delimiter ";"
        temp_record = np.concatenate((outputs[i][:,0], outputs[i][:,1], outputs[i][:,2]), axis=0)
        temp_record = np.reshape(temp_record, (3, len(t)))
        temp_record = np.transpose(temp_record)

        if i == 0:
            record = temp_record
        else:
            record = np.concatenate((record, temp_record), axis=0)
      
    print("Saving the data...")
    np.savetxt(os.path.join(result_dir, str(uid)+"_record_"+str(uid)+".csv"), record, delimiter=",")

    # add ";" to lines with interval len(t)
    with open(os.path.join(result_dir, str(uid)+"_record_"+str(uid)+".csv"), 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if (i % ((len(t))) == 0) and (i != 1):
                lines[i-1] =  lines[i-1] +"timeseries\n"

            else:
                lines[i] = lines[i].strip() + "\n"
    with open(os.path.join(result_dir, str(uid)+"_record_"+str(uid)+".csv"), 'w') as f:
        f.writelines(lines)



    # # save the data
    # print("Saving the data...")

    #        
    # np.savetxt(os.path.join(result_dir, str(uid)+"_Merge4.csv"), Merge4, delimiter=",")
    # np.savetxt(os.path.join(result_dir, str(uid)+"_Merge5.csv"), Merge5, delimiter=",")
    # np.savetxt(os.path.join(result_dir, str(uid)+"_Merge6.csv"), Merge6, delimiter=",")
else:
    pass




# Soft Scatter Plot ===========================================================================================================#
print("Plotting the system...")
plt.figure()
# plt.figure(figsize=(100,100))
ax = plt.axes(projection='3d')


plt.axis('off')
plt.grid(b=None)

if portal_plane:
    x = np.linspace(-0.5, 0.5, 2)
    y = np.linspace(-0.5, 0.5, 2)
    x, y = np.meshgrid(x, y)
    eq = -(a*x)/c - (b*y)/c  - d/c
    ax.plot_surface(x, y, eq, alpha=0.2)




for i in range(num_ic):
    # random 1, 2 or 3
    color_pick = np.random.randint(1, 4)
    if color_pick == 1:
        if color_set == "blue":
            selected_color = "darkslategrey"
        elif color_set == "red":
            selected_color = "darkred"
    elif color_pick == 2:
        if color_set == "blue":
            selected_color = "teal"
        elif color_set == "red":
            selected_color = "maroon"
    elif color_pick == 3:
        if color_set == "blue":
            selected_color = "darkturquoise"
        elif color_set == "red":
            selected_color = "indianred"
    elif color_pick == 4:
        if color_set == "blue":
            selected_color = "lightblue"
        elif color_set == "red":
            selected_color = "firebrick"
    else:
        if color_set == "blue":
            selected_color = "darkslategrey"
        elif color_set == "red":
            selected_color = "darkred"

    # ax.scatter3D(outputs[i][:, 0], outputs[i][:, 1], outputs[i][:, 2],s = 50, alpha = 0.3, color=selected_color, zorder=i, edgecolors='none')
    # ax.scatter3D(outputs[i][:, 0], outputs[i][:, 1], outputs[i][:, 2],s = 50, alpha = 0.3, color=selected_color, zorder=i, edgecolors='none')
    # ax.plot3D(outputs[i][:, 0], outputs[i][:, 1], outputs[i][:, 2], alpha = 0.1, color=selected_color,linewidth=5)
    ax.plot3D(outputs[i][:, 0], outputs[i][:, 1], outputs[i][:, 2], alpha = 1, color=selected_color, linewidth=1, zorder=i)

plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

# ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


if save:
    plt.savefig(os.path.join(result_dir, str(uid)+".png"))
else:
    pass

plt.show()

# ax.scatter3D(out[:, 0], Merge4[:, 1], Merge4[:, 2], alpha = 0.2,color="maroon",)
# ax.scatter3D(Merge5[:, 0], Merge5[:, 1], Merge5[:, 2], alpha = 0.2, color="indianred")
# ax.scatter3D(Merge6[:, 0], Merge6[:, 1], Merge6[:, 2], alpha = 0.2, color="firebrick")





# Plot the system =============================================================================================================#


# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# plt.axis('off')
# plt.grid(b=None)

# if portal_plane:
#     x = np.linspace(-0.5, 0.5, 2)
#     y = np.linspace(-0.5, 0.5, 2)
#     x, y = np.meshgrid(x, y)
#     eq = -(a*x)/c - (b*y)/c  - d/c
#     ax.plot_surface(x, y, eq, alpha=0.2)


# # ax.plot3D(Merge4[:, 0], Merge4[:, 1], Merge4[:, 2], color="black")
# # ax.plot3D(Merge5[:, 0], Merge5[:, 1], Merge5[:, 2], color="lightslategray")
# # ax.plot3D(Merge6[:, 0], Merge6[:, 1], Merge6[:, 2], color="darkslategray")

# ax.scatter3D(Merge4[:, 0], Merge4[:, 1], Merge4[:, 2], alpha = 0.2,color="maroon",)
# ax.scatter3D(Merge5[:, 0], Merge5[:, 1], Merge5[:, 2], alpha = 0.2, color="indianred")
# ax.scatter3D(Merge6[:, 0], Merge6[:, 1], Merge6[:, 2], alpha = 0.2, color="firebrick")

# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5, 0.5)
# ax.set_zlim(-0.5, 0.5)

# # ax.scatter3D(Merge[:, 0], Merge[:, 1], Merge[:, 2], color="green")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')


# if save:
#     plt.savefig(os.path.join(result_dir, str(uid)+".png"))
# else:
#     pass

# plt.show()


# Well thats all folks ========================================================================================================#
# 
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
#
#______________________________________________________________________________________________________________________________#



