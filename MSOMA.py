import numpy as np
import random
import math
import time

# optimized function
def f(x):
    # min=-1, x,y(-100,100), x=-pi y=pi
    #return -math.cos(x[0])*math.cos(x[1])*math.exp(-1*(((x[0] - math.pi)**2)+((x[1] - math.pi)**2)))

    # min=-106.7645, x,y(-2pi,2pi), x=-1.5708 y=-3.1416 or x=4.7124 y=3.1416
    return math.sin(x[0])*math.exp((1-math.cos(x[1]))**2)+math.cos(x[1])*math.exp((1-math.sin(x[0]))**2) + ((x[0]-x[1])**2)

    # min=0, x,y(-5,5), x=0 y=0
    #return 2*(x[0]**2)-1.05*(x[0]**4)+(1/6)*(x[0]**6)+x[0]*x[1]+(x[1]**2)

    # min=-20, x,y(-5,5), x=0 y=0
    #return math.e - 20 * math.exp(-math.sqrt(((x[0] ** 2) + (x[1] ** 2)) / 50)) - math.exp(
    #    0.5 * (math.cos(2 * math.pi * x[0]) + math.cos(2 * math.pi * x[1])))

    # min=3, x,y(-2,2), x=0 y=-1
    #return (1 + ((x[0] + x[1] + 1) ** 2) * (19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * (
    #            30 + ((2 * x[0] - 3 * x[1]) ** 2) * (18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2)))


# population creation (создание популяции)
def pop_create(pop_size, num_of_x, a, b):
    pop = np.empty((num_of_x, pop_size))
    for j in range(pop_size):
        for i in range(num_of_x):
            pop[i][j] = a[i] + random.random() * (b[i] - a[i])
    return pop


# sorting by cost function (сортировка по функции присособленности)
def cost_function_sort(pop, pop_size, num_of_x):
    newpop = np.empty((num_of_x, pop_size))
    cost = {}
    for j in range(pop_size):
        cost[j] = f(pop[:, j])
    list_cost = list(cost.items())
    list_cost.sort(key=lambda i: i[1])
    for j in range(pop_size):
        newpop[:, j] = pop[:, list_cost[j][0]]
    return newpop


# creating a PRT vector (создание вектора PRT)
def create_prt_vector(prt, pop_size, num_of_x):
    prt_vector = np.zeros((num_of_x, pop_size))
    for j in range(pop_size):
        for i in range(num_of_x):
            if random.random() < prt:
                prt_vector[i][j] = 1
    return prt_vector


# movement of individuals towards leaders
def movement(pop_size, prt, num_of_x, NStep, new_pop):

    vector_of_steps_1 = np.zeros((num_of_x, NStep * 4))
    vector_of_steps_2 = np.zeros((num_of_x, NStep * 2))
    vector_of_steps_3 = np.zeros((num_of_x, NStep))

    PRT = create_prt_vector(prt, pop_size * 3, num_of_x)

    # movement to the first leader
    for i in range(pop_size):
        for s in range(4 * NStep):
            vector_of_steps_1[:, s] = new_pop[:, i] + ((new_pop[:, 0] - new_pop[:, i])/(2*NStep)) * (PRT[:, i])*s
        new_pop[:, i] = cost_function_sort(vector_of_steps_1, 4 * NStep, num_of_x)[:, 0]

    # movement to the second leader
    for i in range(pop_size, pop_size * 2):
        for s in range(2 * NStep):
            vector_of_steps_2[:, s] = new_pop[:, i] + ((new_pop[:, pop_size+1] - new_pop[:, i])/NStep) * (PRT[:, i])*s
        new_pop[:, i] = cost_function_sort(vector_of_steps_2, 2 * NStep, num_of_x)[:, 0]

    # movement to the third leader
    for i in range(pop_size * 2, pop_size * 3):
        for s in range(NStep):
            vector_of_steps_3[:, s] = new_pop[:, i] + ((new_pop[:, 2*pop_size+2] - new_pop[:, i])/(NStep/2)) * (PRT[:, i])*s
        new_pop[:, i] = cost_function_sort(vector_of_steps_3, NStep, num_of_x)[:, 0]

    # catching value out of range
    for j in range(pop_size):
        for i in range(num_of_x):
            if new_pop[i][j] < a[i]:
                new_pop[i][j] = a[i]
            if new_pop[i][j] > b[i]:
                new_pop[i][j] = b[i]
    return new_pop


# refinement function (функция уточнения)
def elaboration(pop, NStep, prt, num_of_x):
    NStep *= 10
    PRT = create_prt_vector(prt, 3, num_of_x)
    vector_of_steps = np.zeros((num_of_x, NStep))
    ans = np.zeros((num_of_x, 3))
    for i in range(3):
        for s in range(NStep):
            vector_of_steps[:, s] = pop[:, i] + ((pop[:, 0] - pop[:, i])/(NStep/2)) * (PRT[:, i])*s
        ans[:, i] = cost_function_sort(vector_of_steps, NStep, num_of_x)[:, 0]
    ans = cost_function_sort(ans, 3, num_of_x)
    return ans[:, 0]


# MSOMA funcktion
def MSOMA_count(pop_size, num_of_x, a, b, NStep, prt, Migrations, MinDist):
    start_time = time.time()

    pop = pop_create(pop_size, num_of_x, a, b)
    new_pop = cost_function_sort(pop, pop_size, num_of_x)

    # cloning
    new1_pop = np.concatenate((new_pop, new_pop), axis=1)
    new_pop = np.concatenate((new1_pop, new_pop), axis=1)

    MCount = 0
    # migration loop
    while MCount < Migrations and math.sqrt((1/2) * ((f(new_pop[:, 1]) - f(new_pop[:, 0]))**2 + (f(new_pop[:, 2]) - f(new_pop[:, 0]))**2)) >= MinDist:
        new_pop = movement(pop_size, prt, num_of_x, NStep, new_pop)
        new_pop = cost_function_sort(new_pop, 3 * pop_size, num_of_x)

        # population refreshing
        new_pop = new_pop[:, :round(pop_size*(2/3))]
        new_pop = np.concatenate((new_pop, pop_create(round(pop_size*(1/3)), num_of_x, a, b)), axis=1)
        new_pop = cost_function_sort(new_pop, pop_size, num_of_x)

        # cloning
        new1_pop = np.concatenate((new_pop, new_pop), axis=1)
        new_pop = np.concatenate((new1_pop, new_pop), axis=1)

        MCount += 1

    last_pop = new_pop[:, :3]

    ans = elaboration(last_pop, NStep, prt, num_of_x)

    return ans, f(ans), MCount, time.time() - start_time


# setting paarameters
pop_size = 30
num_of_x = 2
prt = 0.7
NStep = 20
Migrations = 40
MinDist = 0.000001
a = [-2*math.pi, -2*math.pi]
b = [2*math.pi, 2*math.pi]

# x*, f(x*), number of loops, time = MSOMA function (population size, number of variables, constraints a, constraints b, number of steps, prt, number of migrations, minimum distance)
x, f_x, loop, time_run = MSOMA_count(pop_size, num_of_x, a, b, NStep, prt, Migrations, MinDist)

print("min f(x) = ", np.round(f(x), 8))
print("x* = ", np.round(x, 8))
print("loop = ", loop)
print("--- run time %s seconds ---" % time_run)