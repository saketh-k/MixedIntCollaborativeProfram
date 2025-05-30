import sys

n_robots = 2
n_modes = 3
path_length = 1 #m
n_segments = 120
seg_length = 1/n_segments # m

mass_robot1 = 1 #kg
mass_robot2 = 1 #kg

land_efficiency_robot1 = 1
watr_efficiency_robot1 = 3

land_efficiency_robot2 = 3
watr_efficiency_robot2 = 1

# read in input string
input_landscape = sys.argv[2]

resistance_scape_robot1 = [ land_efficiency_robot1\
        if i == "L" else water_efficiency_robot1\
        for i in input_landscape]

resistance_scape_robot2 = [ land_efficiency_robot2\
        if i == "L" else water_efficiency_robot2\
        for i in input_landscape]

# Linear Program Definition:
func_objective = ... # objective function to be minimized

A_ineq = ... # all 4 inequalities to make max fn
b_ineq = np.zeros((2*n_robots,1))

A_eq = ... #rep mat or something
b_eq = np.ones((n_segments,1))

lb = ... # lower bound, all 0 except last one
ub = ... # upper bound, all 1 except last one

int_decision = np.ones((1,n_segments*3))

# Want to add energy term
# Consider how much you need to scale this before it becomes slow
# Send alex working python code and solutions to see if it's right
# Incorporate switching in both and send to alex
# L1, L_inf Switching cost
# Send note with costs and mass and terrain types/ landscapes
solver_options = ...
prob.solve() # lp format for solving problem

# Display
