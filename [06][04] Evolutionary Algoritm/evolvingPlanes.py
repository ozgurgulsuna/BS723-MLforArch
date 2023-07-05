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

import datetime
import os
import json
import random
import copy
import time

import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
from scipy.stats import linregress

from flows import *



# Parameters ==================================================================================================================#

dt = 0.001
t = np.arange(0, dt*5000, dt)

system_A = "Halvorsen"
system_B = "Rossler"

single = 0
smooth = True
portal_plane = True
save = True
print_info = True


# Variables ===================================================================================================================#

num_inds = 20
num_genes = 1
num_generations = 50

tm_size = 5
frac_elites = 0.2
frac_parents = 0.4
mutation_prob = 0.2

print_intervals = 1


# Classes =====================================================================================================================#

class Individual:
    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness

class Gene:
    def __init__(self, r, theta, phi, dir):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.dir = dir
        """dir HAS NO USE FOR NOW"""


# Directory ===================================================================================================================#

filename = "d:/2022-23/Okul/Dersler/BS723/[06][04] Evolutionary Algoritm/chaotic_attractors.json"
result_dir = "d:/2022-23/Okul/Dersler/BS723/[06][04] Evolutionary Algoritm/results/"
output_path = "d:/2022-23/Okul/Dersler/BS723/[06][04] Evolutionary Algoritm/outputs/"+str(num_inds)+"_"+str(num_genes)+"_"+str(tm_size)+"_"+str(frac_elites)+"_"+str(frac_parents)+"_"+str(mutation_prob)+"/"


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


# Random rotation =============================================================================================================#

def plane_rotation(a, b, c, d, r, theta, phi):
    # Define a random rotation matrix
    rotation_r = Rotation.from_rotvec(r* np.array([1, 0.01, 0.01]))
    rotation_theta = Rotation.from_rotvec(theta* np.array([0.01, 1, 0.01]))
    rotation_phi = Rotation.from_rotvec(phi* np.array([0.01, 0.01, 1]))

    # Define a normal vector for the plane
    normal_vector = np.array([a, b, c])

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Rotate the normal vector
    rotated_normal_vector = rotation_r.apply(normal_vector)
    rotated_normal_vector = rotation_theta.apply(rotated_normal_vector)
    rotated_normal_vector = rotation_phi.apply(rotated_normal_vector)

    return rotated_normal_vector[0],rotated_normal_vector[1],rotated_normal_vector[2],d


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
    

def init_population():
    population = []
    for i in range(num_inds):
        genes = []
        for j in range(num_genes):
            r = np.random.rand()*np.pi*2
            theta = np.random.rand()*np.pi*2
            phi = np.random.rand()*np.pi*2
            dir = np.random.randint(0,2)
            genes.append(Gene(r, theta, phi, dir))
        population.append(Individual(genes, 0))
    if print_info:
        print("Population initialized.")
    
    for individual in population:
        evaluate_individual(individual)
    if print_info:
        print("Population evaluated.")

    return population



def evaluate_individual(individual):
    global a,b,c,d
    fitness = 0

    r, theta, phi, dir = individual.genes[0].r, individual.genes[0].theta, individual.genes[0].phi, individual.genes[0].dir 
    a,b,c,d = plane_rotation(0,0,1,0,r,theta,phi)

    fitness = lyap_exp(x0,y0,z0,dt,10)
    if np.isnan(fitness):
        fitness = -200
    individual.fitness = fitness
    return

def tournament_selection(population):
    tournament = []
    candidates = population.copy()
    
    # for high number of individuals, tournament size is limited to the number of individuals
    size = tm_size
    if tm_size > len(candidates):
        size = len(candidates)

    # main tournament loop
    for i in range(size) :
        tournament.append(random.choice(candidates))
        candidates.remove(tournament[i])
    best = tournament[0]
    for individual in tournament:
        if individual.fitness > best.fitness:
            best = individual
    return best

def elitism(population):
    attendees = copy.deepcopy(population)
    elites = []
    best_index = []
    for i in range(int(frac_elites*num_inds)):
        j = 0
        k = 0
        best = attendees[0]
        for individual in attendees:
            if individual.fitness > best.fitness:
                k = j
                best = individual
            j+=1
        best_index.append(k)
        elites.append(best)
        attendees.remove(best)
    # print("best index: ", best_index)
    # for j in sorted(best_index, reverse=True):
    #     del population[j]
    return elites

def natural_selection(population):
    parents = []
    testants = population
    for i in range(int(num_inds - int(frac_elites*num_inds)- int(frac_parents*num_inds))):
        parents.append(tournament_selection(testants))
        testants.remove(parents[i])
        ### population.remove(parents[i])
    return parents

def parent_selection(population):
    parents = []
    testants = population
    for i in range(int(frac_parents*num_inds)):
        parents.append(tournament_selection(testants))
        testants.remove(parents[i])
        ### population.remove(parents[i])
    return parents

# Crossover
def crossover(parents):
    children = []
    for i in range(int(frac_parents*num_inds/2)):
        child1 = []
        child2 = []
        for j in range(num_genes):
            if random.random() < 0.5:
                temp1 = copy.deepcopy(parents[i].genes[j])
                temp1.r = parents[i].genes[j].r
                temp2 = copy.deepcopy(parents[i+1].genes[j])
                temp2.r = parents[i+1].genes[j].r
            else:
                temp1 = copy.deepcopy(parents[i+1].genes[j])
                temp1.r = parents[i+1].genes[j].r
                temp2 = copy.deepcopy(parents[i].genes[j])
                temp2.r = parents[i].genes[j].r
            if random.random() < 0.5:
                temp1.theta = parents[i].genes[j].theta
                temp2.theta = parents[i+1].genes[j].theta
            else:
                temp1.theta = parents[i+1].genes[j].theta
                temp2.theta = parents[i].genes[j].theta
            if random.random() < 0.5:
                temp1.phi = parents[i].genes[j].phi
                temp2.phi = parents[i+1].genes[j].phi
            else:
                temp1.phi = parents[i+1].genes[j].phi
                temp2.phi = parents[i].genes[j].phi
            if random.random() < 0.5:
                temp1.dir = parents[i].genes[j].dir
                temp2.dir = parents[i+1].genes[j].dir
            else:
                temp1.dir = parents[i+1].genes[j].dir
                temp2.dir = parents[i].genes[j].dir
            child1.append(temp1)
            child2.append(temp2)

        children.append(Individual(child1, -10))
        children.append(Individual(child2, -10))
        """-10 fitness means that not evaluated yet"""
    return children


def mutation(population):
    population = copy.deepcopy(population)
    for individual in population:
        individual.fitness = -20 # -20 fitness means that not evaluated yet
        for gene in individual.genes:
            if random.random() < mutation_prob:
                """guided mutation"""
                gene.r = gene.r + random.uniform(-np.pi/8,np.pi/8)
                gene.theta = gene.theta + random.uniform(-np.pi/8,np.pi/8)
                gene.phi = gene.phi + random.uniform(-np.pi/8,np.pi/8)
                # gene.dir = gene.dir + ...... # Lets not mutate the direction for now
    return population


# Evaluate the system =========================================================================================================#

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
        # print("Iteration: ", i)
        traj1 = odeint(merge, (x, y, z), t)
        traj2 = odeint(merge, (x+0.0001, y, z), t)
        if max(traj1[:,0]) > 2 or max(traj1[:,1]) > 2 or max(traj1[:,2]) > 2 or min(traj1[:,0]) < -2 or min(traj1[:,1]) < -2 or min(traj1[:,2]) < -2:
            return -100
        lyap.append(calculate_lyapunov_exponent(traj1, traj2, dt))
        x = traj1[-1, 0]
        y = traj1[-1, 1]
        z = traj1[-1, 2]
        # print("Lyapunov exponent: ", lyap[-1])
    return np.mean(lyap)

# Save & Print The Tag ========================================================================================================#

# date_created = datetime.datetime.now()
# uid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# f = open(logs, "a")
# tag = "================================================================================================================="+"\n"
# tag += "model_A: "+str(model_A)+"\n"
# tag += "model_B: "+str(model_B)+"\n"
# tag += "plane_parameters: a = "+str(a)+" b = "+str(b)+" c = "+str(c)+" d = "+str(d)+"\n"
# tag += "other_parameters: t = "+str(dt*len(t))+", dt = "+str(dt)+", smooth = "+str(smooth)+", saved = "+str(save) +"\n"
# tag += "initial_conditions: x0= "+str(x0)+" y0= "+str(y0)+" z0= "+str(z0)+"\n"
# tag += "initial_conditions: x1= "+str(x1)+" y1= "+str(y1)+" z1= "+str(z1)+"\n"
# tag += "initial_conditions: x2= "+str(x2)+" y2= "+str(y2)+" z2= "+str(z2)+"\n"
# tag += "date_created: "+str(date_created)+"\n"
# tag += "uid: "+str(uid)+"\n"
# # tag += "lyapunov_exponent: "+str(lyap)+"\n"
# tag += "================================================================================================================="+"\n"
# print(tag)
# f.write(tag)
# f.close()




# Plot the system =============================================================================================================#

# print("Plotting the system...")
# plt.figure()
# ax = plt.axes(projection='3d')

# if portal_plane:
#     x = np.linspace(-0.5, 0.5, 2)
#     y = np.linspace(-0.5, 0.5, 2)
#     x, y = np.meshgrid(x, y)
#     eq = -(a*x)/c - (b*y)/c  - d/c
#     ax.plot_surface(x, y, eq, alpha=0.2)


# ax.plot3D(Merge4[:, 0], Merge4[:, 1], Merge4[:, 2], color="black")
# ax.plot3D(Merge5[:, 0], Merge5[:, 1], Merge5[:, 2], color="lightslategray")
# ax.plot3D(Merge6[:, 0], Merge6[:, 1], Merge6[:, 2], color="darkslategray")

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

def main():
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    init_fit = []
    later_fit = []
    start_time = time.time()
    population = init_population()
    best = population[0]
    best_temp = population[0]
    for i in range(num_generations):
        a = 0
        if print_info == True and i%print_intervals == 0:
            print("START")
        for individual in population:
            a += 1
            evaluate_individual(individual)
            if print_info == True and i%print_intervals == 0:
                print("individual:",a,"fitness:",  individual.fitness)

        for individual in population:
            if (individual.fitness > best.fitness):
                best = individual
        
        if print_info == True and i%print_intervals == 0:
            print("Generation: ", i, "Best fitness: ", best.fitness)

        if i<num_generations:
            init_fit.append(best.fitness)
            plt.plot(init_fit)
            # plt.draw()
            plt.title("Fitness Plot from Generation 1 to 10000")
            plt.ylabel('Fitness')
            plt.xlabel('Generation')
            if i%print_intervals == 0 and save == True:
                plt.savefig(output_path+"mmmmm"+"_fitness.png",dpi=200)
            # plt.pause(0.1)
            plt.clf()
        # if i>1000 and i<10000:
        #     later_fit.append(best.fitness)
        #     plt.plot(later_fit)
        #     # plt.draw()
        #     plt.title("Fitness Plot from Generation 1000 to 10000")
        #     plt.ylabel('Fitness')
        #     plt.xlabel('Generation')
        #     if i%100 == 0 and save == True:
        #         plt.savefig(output_path+"mmmmm"+"_fitness_1000.png",dpi=200)
        #     # plt.pause(0.1)
        #     plt.clf()
        
        
        # Elites selected, isolated from the population
        elites = elitism(population)

        # Parents selected, isolated from the population
        parents = parent_selection(population)

        # Crossover
        children = crossover(parents)

        # Natural selection
        population = natural_selection(population)

        # Mutation
        children = mutation(children)
        population = mutation(population)

        population = elites + children + population

    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    best = population[0]
    for individual in population:
        if individual.fitness > best.fitness:
            best = individual
            print(best.fitness)
    return best




# Run
best_case = main()

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



