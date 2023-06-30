

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

solver = 'euler'
dt = 0.005
t = np.arange(0, 10, dt)

#----------------------------------------------------------#
# TSUCS 1 Attractor (Three-Scroll Unified Chaotic System)
# Initial values
x0 = 0.1
y0 = 0.1
z0 = 0.1

# Parameters
a = 40
c = 0.833
d = 0.5
e = 0.65
f = 20

# Yu-Wang Attractor
# Initial values
x1 = 0.1
y1 = 0.1
z1 = 0.1

# Parameters
a1 = 10
b1 = 40
c1 = 2
d1 = 2.5


#----------------------------------------------------------#
# Define the systems
def TSUCS1(state, t):
    x, y, z = state
    return a * (y - x) + d * x * z, f * y - x * z , c * z + x * y - e * x**2


def YuWang(state, t):
    x, y, z = state
    return a1*(y-x), b1*x-c1*x*z, np.exp(x*y)-d1*z

#----------------------------------------------------------#
# Solve the systems
TSUCS1 = odeint(TSUCS1, (x0, y0, z0), t)

YuWang = odeint(YuWang, (x1, y1, z1), t)

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

#----------------------------------------------------------#

# Plot the system
plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(YuWang[:, 0], YuWang[:, 1], YuWang[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#----------------------------------------------------------#

# TSUCS 1 Attractor (Three-Scroll Unified Chaotic System)

#----------------------------------------------------------#

# Plot the system

fig = plt.figure()







