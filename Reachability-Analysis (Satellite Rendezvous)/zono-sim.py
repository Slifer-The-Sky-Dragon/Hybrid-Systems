import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.linalg import expm
import random

ZONO_C_X = -900
ZONO_C_Y = -400
ZONO_C_VX = 0
ZONO_C_VY = 0

STEP_SIZE = 1
MAX_TIME = 270

START_STATE = -1
FAR_MODE = 0
NEAR_MODE = 1
PASSIVE_MODE = 2

NEAR_THRESHOLD = -100.0
PASSIVE_MIN_TIME_STEP = 100
PASSIVE_MAX_TIME_STEP = 120


def get_solution_mats_sim(step_size):
    'get far, near, and passive solution matrices'

    rv = []

    # far
    a_mat = np.array([[0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0], 
                      [-0.057599765881773, 0.000200959896519766, -2.89995083970656, 0.00877200894463775], 
                      [-0.000174031357370456, -0.0665123984901026, -0.00875351105536225, -2.90300269286856]])

    rv.append(expm(a_mat * step_size))

    # near
    a_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [-0.575999943070835, 0.000262486079431672, -19.2299795908647, 0.00876275931760007],
                      [-0.000262486080737868, -0.575999940191886, -0.00876276068239993, -19.2299765959399]])

    rv.append(expm(a_mat * step_size))

    # passive
    a_mat = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0.0000575894721132000, 0, 0, 0.00876276],
                      [0, 0, -0.00876276, 0]])

    rv.append(expm(a_mat * step_size))

    return rv


class Zonotope:
	def __init__(self, center, generators):
		self.center = center.copy()
		self.generators = generators.copy()

	def maximize(self, direction):
		result = self.center.copy()

		for generator in self.generators.T:
			if np.dot(generator, direction) > 0:
				result += generator.reshape(len(generator), 1)
			else:
				result -= generator.reshape(len(generator), 1)

		return result

	def linear_transformation(self, A):
		self.center = A @ self.center
		self.generators = A @ self.generators


	def copy(self):
		return Zonotope(self.center, self.generators)

	def __str__(self):
		return f"Center:\n{self.center}\nGenerators:\n{self.generators}"

def plot_zonotope(zonotope, mode):
	xs = []
	ys = []

	if mode == START_STATE:
		color = "r-"
		lw = 1
	elif mode == FAR_MODE:
		color = "b:"
		lw = 1
	elif mode == NEAR_MODE:
		color = "y-o"
		lw = 2
	elif mode == PASSIVE_MODE:
		color = "m-"
		lw = 2

	for theta in np.linspace(0, 2 * np.pi, 100):
		direction_vector = np.array([[np.cos(theta)], [np.sin(theta)], [0], [0]])
		zono_max_point = zonotope.maximize(direction_vector)

		xs.append(zono_max_point[0])
		ys.append(zono_max_point[1])

	xs.append(xs[0])
	ys.append(ys[0])

	plt.plot(xs, ys, color, lw)

def get_initial_zonotop_params():
	zono_center = np.array([[ZONO_C_X], [ZONO_C_Y], [ZONO_C_VX], [ZONO_C_VY]])

	zono_generator1 = np.array([[25], [0], [0], [0]])
	zono_generator2 = np.array([[0], [25], [0], [0]])
	zono_generators = np.concatenate((zono_generator1, zono_generator2), axis=1)

	return zono_center, zono_generators

def perform_rechability_analysis(start_zonotope):
	plot_zonotope(start_zonotope, mode=START_STATE)
	A_matrices = get_solution_mats_sim(STEP_SIZE)

	matrix_multiplication_cnt = 0

	queue = [(start_zonotope, FAR_MODE, 0)]

	while len(queue) != 0:
		cur_state_zonotope, cur_mode, cur_time_step = queue.pop(0)

		if cur_time_step >= MAX_TIME or (cur_time_step >= PASSIVE_MAX_TIME_STEP and cur_mode != PASSIVE_MODE):
			continue

		plot_zonotope(cur_state_zonotope, mode=cur_mode)

		if cur_mode == FAR_MODE:
			cur_state_zonotope.linear_transformation(A_matrices[FAR_MODE])
			matrix_multiplication_cnt += 1

			zonotope_max_value = cur_state_zonotope.maximize([[1], [0], [0], [0]])[0]
			zonotope_min_value = cur_state_zonotope.maximize([[-1], [0], [0], [0]])[0]

			if zonotope_min_value < NEAR_THRESHOLD:
				queue.append((cur_state_zonotope, FAR_MODE, cur_time_step + 1))
			if zonotope_max_value >= NEAR_THRESHOLD:
				queue.append((cur_state_zonotope.copy(), NEAR_MODE, cur_time_step + 1))
			if cur_time_step >= PASSIVE_MIN_TIME_STEP and cur_time_step <= PASSIVE_MAX_TIME_STEP:
				queue.append((cur_state_zonotope.copy(), PASSIVE_MODE, cur_time_step + 1))

		elif cur_mode == NEAR_MODE:
			cur_state_zonotope.linear_transformation(A_matrices[NEAR_MODE])
			matrix_multiplication_cnt += 1
			queue.append((cur_state_zonotope, NEAR_MODE, cur_time_step + 1))

			if cur_time_step >= PASSIVE_MIN_TIME_STEP and cur_time_step <= PASSIVE_MAX_TIME_STEP:
				queue.append((cur_state_zonotope.copy(), PASSIVE_MODE, cur_time_step + 1))

		elif cur_mode == PASSIVE_MODE:
			cur_state_zonotope.linear_transformation(A_matrices[PASSIVE_MODE])
			matrix_multiplication_cnt += 1

			queue.append((cur_state_zonotope, PASSIVE_MODE, cur_time_step + 1))
	print(matrix_multiplication_cnt)

def main():
	zono_center, zono_generators = get_initial_zonotop_params()

	start_zonotope = Zonotope(zono_center, zono_generators)

	perform_rechability_analysis(start_zonotope)

####End of my Reachability Analysis code#####

#########################SIM Satellite code to plot the simulations#########################

def step(state, mode, sol_mats):
    'use matrix exponential solution to find next state'

    if mode == 'far':
        sol_mat = sol_mats[0]
    elif mode == 'near':
        sol_mat = sol_mats[1]
    elif mode == 'passive':
        sol_mat = sol_mats[2]
        
    new_state = sol_mat.dot(state)

    return new_state

def simulate(start_x, start_y, passive_time):
    '''simulate and plot the satellite rendezvous system
    '''
    
    tmax = 270 # minutes
    step_size = 1.0
    num_steps = int(round(tmax / step_size))
    
    passive_step = int(round(passive_time / step_size))

    # x, y, vx, vy
    cur_state = np.array([[start_x], [start_y], [0], [0]])
    cur_mode = 'far'

    sol_mats = get_solution_mats(step_size)

    xs = [start_x]
    ys = [start_y]

    for cur_step in range(num_steps):
        cur_state = step(cur_state, cur_mode, sol_mats)
        assert cur_state.shape == (4, 1)

        xs.append(cur_state[0, 0])
        ys.append(cur_state[1, 0])
        prev_mode = cur_mode

        # check guards
        if cur_mode != 'passive' and cur_step + 1 == passive_step:
            # next step should be in passive mode
            cur_mode = 'passive'
        elif cur_mode == 'far' and cur_state[0] >= -100:
            cur_mode = 'near'

        if prev_mode != cur_mode or cur_step == num_steps - 1:
            # changed mode or reached end of sim, plot now

            if prev_mode == 'far':
                color = 'lime'
                zorder=0
            elif prev_mode == 'near':
                color = 'orange'
                zorder=1
            elif prev_mode == 'passive':
                color = 'c:'
                zorder=0

            plt.plot(xs, ys, color, lw=1, alpha=0.5, zorder=zorder)
            xs = [cur_state[0, 0]]
            ys = [cur_state[1, 0]]

def get_solution_mats(step_size):
    'get far, near, and passive solution matrices'

    rv = []

    # far
    a_mat = np.array([[0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0], 
                      [-0.057599765881773, 0.000200959896519766, -2.89995083970656, 0.00877200894463775], 
                      [-0.000174031357370456, -0.0665123984901026, -0.00875351105536225, -2.90300269286856]])

    rv.append(expm(a_mat * step_size))

    # near
    a_mat = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [-0.575999943070835, 0.000262486079431672, -19.2299795908647, 0.00876275931760007],
                      [-0.000262486080737868, -0.575999940191886, -0.00876276068239993, -19.2299765959399]])

    rv.append(expm(a_mat * step_size))

    # passive
    a_mat = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0.0000575894721132000, 0, 0, 0.00876276],
                      [0, 0, -0.00876276, 0]])

    rv.append(expm(a_mat * step_size))

    return rv
    
def init_plot():
    'plot the background and return axis object'

    # set plot style
    plt.style.use('bmh')
    plt.rcParams.update({
        'font.family': 'serif',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 20,
        'axes.titlesize': 28,
        'path.simplify': False
    })


    plt.ylabel('Y')
    plt.xlabel('X')

    plt.title('Satellite Rendezvous')

    plt.xlim([-950, 300])
    plt.ylim([-450, 300])

    y = 57.735
    line = [(-100.0, y), (-100.0, -y), (0.0, 0.0), (-100.0, y)]
    c1 = collections.LineCollection([line], colors=('gray'), linewidths=2, linestyle='dashed')
    plt.gca().add_collection(c1)

    rad = 5
    line = [(-rad, -rad), (-rad, rad), (rad, rad), (rad, -rad), (-rad, -rad)]
    c2 = collections.LineCollection([line], colors=('red'), linewidths=2)
    plt.gca().add_collection(c2)


def plot_box(box, color):
    'plot a box'

    xmin, xmax = box[0]
    ymin, ymax = box[1]

    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]

    plt.plot(xs, ys, color)

def main2():
    'main entry point'

    random.seed(0) # deterministic random numbers
    init_box = [(-925.0, -875.0), (-425.0, -375.0)]
    
    init_plot()
    #plot_box(init_box, 'k-')

    # simulate 100 times
    for _ in range(100):
        start_x = random.uniform(*init_box[0])
        start_y = random.uniform(*init_box[1])
        passive_time = random.randint(100, 120)

        simulate(start_x, start_y, passive_time)

    # save plot
    plt.savefig('sim.png')

    # zoom in and re-plot
    ax = plt.gca()
    ax.set_xlim([-110, 20])
    ax.set_ylim([-70, 60])
    plt.savefig('zoomed_sim.png')


if __name__ == "__main__":
	#my code
	main()
	
	# sim_satellite code execution
	main2()

