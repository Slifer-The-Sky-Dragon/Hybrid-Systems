import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import copy

def ReLU(input_vec):
	activation_pattern = ""
	result = input_vec.copy()

	n = input_vec.shape[0]

	for i in range(n):
		if result[i][0] < 0:
			activation_pattern += "-"
			result[i][0] = 0.0
		else:
			activation_pattern += "+"

	return result, activation_pattern

def neural_network(inputs):
	W1 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
	b1 = np.array([[0], [0]])

	z1 = W1 @ inputs + b1
	a1, l1_activation_pattern = ReLU(z1)

	W2 = np.array([[1, 0], [0, 1]])
	b2 = np.array([[0], [-np.sqrt(2) / 2]])

	z2 = W2 @ a1 + b2

	y, l2_activation_pattern = ReLU(z2)

	netwrok_activate_pattern = l1_activation_pattern + l2_activation_pattern

	return y, netwrok_activate_pattern

def q3_simulation():
	unique_activations = {}
	color_palette = ["bo", "ro", "go", "mo", "co", "ko"]

	color_x1s = {}
	color_x2s = {}
	color_id2activation = {}

	for t in range(500):
		x1 = random.uniform(0.5, 1)
		x2 = random.uniform(0, 2)
		x = np.array([[x1], [x2]])
		y, activation_pattern = neural_network(x)

		if activation_pattern not in unique_activations.keys():
			unique_activations[activation_pattern] = len(unique_activations.keys())
			color_id2activation[unique_activations[activation_pattern]] = activation_pattern
			color_x1s[unique_activations[activation_pattern]] = [x1]
			color_x2s[unique_activations[activation_pattern]] = [x2]
		else:
			color_x1s[unique_activations[activation_pattern]].append(x1)
			color_x2s[unique_activations[activation_pattern]].append(x2)

	for color_id in range(len(unique_activations.keys())):
		color = color_palette[color_id]
		plt.plot(color_x1s[color_id], color_x2s[color_id], color, label=color_id2activation[color_id])		

	plt.xlim(-2.5, 2.5)
	plt.ylim(-2.5, 2.5)
	plt.legend()
	plt.show()

def plot_star(axis, star, color, x_label):
	xs = []
	ys = []

	for theta in np.linspace(0, 2 * np.pi, 100):
		direction_vector = np.array([[np.cos(theta)], [np.sin(theta)]])
		star_max_point = star.maximize(direction_vector)

		xs.append(star_max_point[0])
		ys.append(star_max_point[1])

	xs.append(xs[0])
	ys.append(ys[0])

	axis.plot(xs, ys, color, zorder=1)
	axis.axvline(x=0, color='k', linestyle='--', linewidth=1, zorder=0)
	axis.axhline(y=0, color='k', linestyle='--', linewidth=1, zorder=0)

	axis.set_xlim(-2.5, 2.5)
	axis.set_ylim(-2.5, 2.5)
	axis.set_xlabel(x_label)

def plot_star_domain(axis, star, color, x_label):
	xs = []
	ys = []

	for theta in np.linspace(0, 2 * np.pi, 100):
		direction_vector = np.array([[np.cos(theta)], [np.sin(theta)]])

		min_direction = direction_vector * -1

		domain_max_point = linprog(c=min_direction, A_ub=star.G, b_ub=star.g, bounds=(None, None)).x
		domain_max_point = domain_max_point.reshape(domain_max_point.shape[0], 1)

		xs.append(domain_max_point[0])
		ys.append(domain_max_point[1])

	xs.append(xs[0])
	ys.append(ys[0])

	axis.plot(xs, ys, color, zorder=1)
	axis.axvline(x=0, color='k', linestyle='--', linewidth=1, zorder=0)
	axis.axhline(y=0, color='k', linestyle='--', linewidth=1, zorder=0)

	axis.set_xlim(-2.5, 2.5)
	axis.set_ylim(-2.5, 2.5)
	axis.set_xlabel(x_label)

class Star:
	def __init__(self, center, generators, G, g, negative_dim=None):
		self.center = center.copy()
		self.generators = generators.copy()
		self.G = G.copy()
		self.g = g.copy()
		self.negative_dim = negative_dim

	def affine_step(self, step_mat, bias_vec):
		self.center = step_mat @ self.center + bias_vec
		self.generators = step_mat @ self.generators

	def maximize(self, max_direction):
		min_direction = max_direction * -1

		domain_direction = min_direction.T @ self.generators
		domain_max_point = linprog(c=domain_direction, A_ub=self.G, b_ub=self.g, bounds=(None, None)).x

		range_point = self.center + self.generators @ domain_max_point.reshape(domain_max_point.shape[0], 1)

		return range_point

	def intersect_halfspace(self, new_condition, new_g):
		domain_condition = new_condition @ self.generators
		domain_g = new_g - new_condition @ self.center

		self.G = np.concatenate((self.G, domain_condition))
		self.g = np.concatenate((self.g, domain_g))

	def is_splittable(self, dim_ind):
		max_direction = np.zeros(self.center.shape)
		max_direction[dim_ind][0] = 1

		min_direction = max_direction * -1

		max_point = self.maximize(max_direction)[dim_ind][0]
		min_point = self.maximize(min_direction)[dim_ind][0]

		if max_point > 0.0 and min_point < 0.0 and abs(max_point) > 1e-6 and abs(min_point) > 1e-6:
			return True
		return False		

	def split_over_dim(self, dim_ind):
		if self.is_splittable(dim_ind):
			range_condition = np.zeros(self.center.shape)
			range_condition[dim_ind][0] = -1

			positive_part_star = self.copy()
			negative_part_star = self.copy()

			positive_part_star.intersect_halfspace(range_condition.T, 0)
			negative_part_star.intersect_halfspace(-1 * range_condition.T, 0)

			return positive_part_star, negative_part_star

		return -1

	def project_to_dim(self, dim_ind):
		self.center[dim_ind][0] = 0
		for j in range(self.generators.shape[1]):
			self.generators[dim_ind][j] = 0

	def copy(self):
		return Star(self.center, self.generators, self.G, self.g)

def plot_states_info(cur_states_stars, title):
	color_palette = ["b-o", "r-", "g--", "y-", "c-", "m-"]
	fig, axis = plt.subplots(1, 2)
	for i in range(len(cur_states_stars)):
		cur_state_star = cur_states_stars[i][0]
		plot_star_domain(axis[0], cur_state_star, color_palette[i], "Star Domain")
		plot_star(axis[1], cur_state_star, color_palette[i], "Star Range")
	fig.suptitle(title)
	plt.show()

def q4_reachability_analysis():
	start_center = np.array([[0.75], [1]])
	start_generators = np.array([[0.25, 0], [0, 1]])
	start_G = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
	start_g = np.array([[1], [1], [1], [1]])
	start_star = Star(start_center, start_generators, start_G, start_g)

	prev_states_stars = []
	cur_states_stars = [[start_star, []]]

	#first affine transformation
	W1 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
	b1 = np.array([[0], [0]])

	prev_states_stars = copy.deepcopy(cur_states_stars)
	cur_states_stars = []
	for prev_state_star_info in prev_states_stars:
		new_state_star = prev_state_star_info[0].copy()
		negative_dim = copy.deepcopy(prev_state_star_info[1])
		new_state_star.affine_step(W1, b1)
		cur_states_stars.append([new_state_star, negative_dim])

	plot_states_info(cur_states_stars, title="The affine transformation before the first hidden layer")

	#first ReLU Spliting
	for dim_ind in range(2):
		prev_states_stars = copy.deepcopy(cur_states_stars)
		cur_states_stars = []
		for prev_state_star_info in prev_states_stars:
			prev_state_star = prev_state_star_info[0]
			if prev_state_star.is_splittable(dim_ind=dim_ind):
				splitted_positive_star, splitted_negative_star = prev_state_star.split_over_dim(dim_ind=dim_ind)
				
				splitted_positive_star_negative_dim = copy.deepcopy(prev_state_star_info[1])
				splitted_negative_star_negative_dim = copy.deepcopy(prev_state_star_info[1])
				splitted_negative_star_negative_dim.append(dim_ind)

				cur_states_stars.append([splitted_positive_star, splitted_positive_star_negative_dim])
				cur_states_stars.append([splitted_negative_star, splitted_negative_star_negative_dim])
			else:
				cur_states_stars.append(prev_state_star_info)

	plot_states_info(cur_states_stars, title="The splitting the set after first ReLU")

	#First Projection
	prev_states_stars = copy.deepcopy(cur_states_stars)
	cur_states_stars = []
	for prev_state_star_info in prev_states_stars:
		new_state_star = prev_state_star_info[0].copy()
		for dim_ind in prev_state_star_info[1]:
			new_state_star.project_to_dim(dim_ind)
		cur_states_stars.append([new_state_star, []])	

	plot_states_info(cur_states_stars, title="First Projection of Negative Inputs")

	#Second Affine Transformation
	W2 = np.array([[1, 0], [0, 1]])
	b2 = np.array([[0], [-np.sqrt(2) / 2]])

	prev_states_stars = copy.deepcopy(cur_states_stars)
	cur_states_stars = []
	for prev_state_star_info in prev_states_stars:
		new_state_star = prev_state_star_info[0].copy()
		negative_dim = copy.deepcopy(prev_state_star_info[1])
		new_state_star.affine_step(W2, b2)
		cur_states_stars.append([new_state_star, negative_dim])

	plot_states_info(cur_states_stars, title="The affine transformation before the second hidden layer")

	#Second ReLU Spliting
	for dim_ind in range(2):
		prev_states_stars = copy.deepcopy(cur_states_stars)
		cur_states_stars = []
		for prev_state_star_info in prev_states_stars:
			prev_state_star = prev_state_star_info[0]
			if prev_state_star.is_splittable(dim_ind=dim_ind):
				splitted_positive_star, splitted_negative_star = prev_state_star.split_over_dim(dim_ind=dim_ind)
				
				splitted_positive_star_negative_dim = copy.deepcopy(prev_state_star_info[1])
				splitted_negative_star_negative_dim = copy.deepcopy(prev_state_star_info[1])
				splitted_negative_star_negative_dim.append(dim_ind)

				cur_states_stars.append([splitted_positive_star, splitted_positive_star_negative_dim])
				cur_states_stars.append([splitted_negative_star, splitted_negative_star_negative_dim])
			else:
				cur_states_stars.append(prev_state_star_info)

	plot_states_info(cur_states_stars, title="The splitting the set after second ReLU")

	#Second Projection
	prev_states_stars = copy.deepcopy(cur_states_stars)
	cur_states_stars = []
	for prev_state_star_info in prev_states_stars:
		new_state_star = prev_state_star_info[0].copy()
		for dim_ind in prev_state_star_info[1]:
			new_state_star.project_to_dim(dim_ind)
		cur_states_stars.append([new_state_star, []])	

	plot_states_info(cur_states_stars, title="Second Projection of Negative Inputs\n Final Reachablity Analysis")



if __name__ == "__main__":
	# Question 1 and 2
	# x1, x2 = map(float, input("Enter the inputs x1, x2: ").split())
	# x = np.array([[x1], [x2]])
	# y, activation_pattern = neural_network(x)
	# print(f"Output: {y}\nActivate Pattern: {activation_pattern}")

	# Question 3
	# q3_simulation()

	# Question 4
	q4_reachability_analysis()
