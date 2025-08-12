from scipy.integrate import RK45
import math
import matplotlib.pyplot as plt
import numpy as np

M = 1.0
L = 1.0
G = 9.81

def pendulum_derivative_function(t, pend_state):
	pend_phi = pend_state[0]
	pend_v = pend_state[1]

	phi_der = pend_v
	v_der = -(G / L) * math.sin(pend_phi)

	return np.array([phi_der, v_der])

cur_v = 0
cur_phi = 3
initial_state = np.array([cur_phi, cur_v])

cur_t = 0
t_max = 10

atol = 1e-8
rtol = 1e-5

rk45 = RK45(pendulum_derivative_function, 0, initial_state, t_max, max_step=0.01, rtol=rtol, atol=atol)

times = [0]
output_phi = [cur_phi]

while cur_t < t_max:
	rk45.step()

	times.append(rk45.t)
	output_phi.append(rk45.y[0])

	cur_t = rk45.t
	print(cur_t)

plt.plot(times, output_phi)
plt.title("Pendulum Model $\\phi$ Output")
plt.xlabel("t")
plt.ylabel("$\\phi$")
plt.show()





