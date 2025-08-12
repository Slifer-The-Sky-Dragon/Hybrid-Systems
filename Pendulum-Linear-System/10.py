from scipy.integrate import RK45
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

M = 1.0
L = 1.0
G = 9.81


SATURATION_POINT = 4.0

KP = 11.81
KD = 3

def pendulum_nonlinear_derivative_function(t, pend_state):
	pend_phi = pend_state[0]
	pend_v = pend_state[1]

	r = np.array([[0], [0]])
	s = np.array([[pend_phi - math.pi], [pend_v]])

	u = np.matmul(np.array([KP, KD]).reshape(1, 2), (r - s))[0][0]

	if u >= 0:
		u = min(u, SATURATION_POINT)
	else:
		u = max(u, -SATURATION_POINT)

	phi_der = pend_v
	v_der = -(G / L) * math.sin(pend_phi) + (u / (M * L * L))

	return np.array([phi_der, v_der])

def generate_initial_states(center, r):
	result = []
	for i in range(36):
		rotation = i * (math.pi / 18)
		new_x = center[0] + r * math.cos(rotation)
		new_y = center[1] + r * math.sin(rotation)
		result.append(np.array([new_x, new_y]))
	return result

def plot_initial_states(all_initial_states):
	point_x = []
	point_y = []
	for state in all_initial_states:
		point_x.append(state[0])
		point_y.append(state[1])

	plt.plot(point_x, point_y, 'bo')

def simulate(derivative_function, initial_state):
	cur_t = 0
	t_max = 2

	atol = 1e-8
	rtol = 1e-5

	rk45 = RK45(derivative_function, 0, initial_state, t_max, max_step=0.01, rtol=rtol, atol=atol)

	output_phi = [initial_state[0]]
	state_v = [initial_state[1]]

	while cur_t < t_max:
		rk45.step()

		output_phi.append(rk45.y[0])
		state_v.append(rk45.y[1])

		cur_t = rk45.t
		print(cur_t)

	return output_phi, state_v

for step in range(1, 16):
	radius = step / 10.0
	cur_v = 0
	cur_phi = math.pi
	center = np.array([cur_phi, cur_v])
	all_initial_states = generate_initial_states(center, radius)
	plot_initial_states(all_initial_states)

	for cur_initial_state in all_initial_states:
		output_phi, state_v = simulate(pendulum_nonlinear_derivative_function, cur_initial_state)
		plt.plot(output_phi, state_v, 'b-')

	nonlinear_handle = Line2D([0], [0], color='blue', label="nonlinear")

	plt.title(f"Result for r = {radius}")
	plt.legend(handles=[nonlinear_handle], loc="upper left")
	plt.xlabel("Angle $\\phi$")
	plt.ylabel("Angular Velocity v")
	plt.show()
