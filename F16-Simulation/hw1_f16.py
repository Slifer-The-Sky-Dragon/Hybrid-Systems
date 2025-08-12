'''
Hybrid Systems homework 1

F-16 simulation
'''

import time

import numpy as np
import matplotlib.pyplot as plt

import aerobench.plot as plot
from aerobench.waypoint_autopilot import WaypointAutopilot
from aerobench.util import StateIndex
from scipy.integrate import RK45


PART3 = False
PART4 = True

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = np.deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # add three states for the low-level controller integrators
    init += [0, 0, 0]
    
    tmax = 150 # simulation time

    # list of waypoints
    waypoints = [[-5000.0, -7500.0, alt],
                 [-15000.0, -7500.0, alt],
                 [-20000.0, 0.0, alt+500.0],
                 [0.0, 15000.0, alt]]

    step_size = 0.05

    if PART3:
        part3_simulations(init, tmax, step_size, waypoints)
        return

    start_time = time.perf_counter()

    #Different Simulation Methods

    #states = run_euler_f16_sim(init, tmax, step_size, waypoints)
    if PART4 == True:
        states = run_rk45_f16_sim(init, tmax, step_size, waypoints)
    else:
        states = run_rk4_classic_f16_sim(init, tmax, step_size, waypoints)

    ############

    runtime = time.perf_counter() - start_time
    print(f"Simulation Completed in {round(runtime, 2)} seconds")

    # print final state
    final_state = states[-1]
    x = final_state[StateIndex.POS_E]
    y = final_state[StateIndex.POS_N]
    z = final_state[StateIndex.ALT]
    
    #Different Simulation Outputs

    #print(f"Euler with step size: {step_size}, final state x, y, z: {round(x, 3)}, {round(y, 3)}, {round(z, 3)}")
    print(f"RK4 with step size: {step_size}, final state x, y, z: {round(x, 3)}, {round(y, 3)}, {round(z, 3)}")

    # plot
    plot.plot_overhead(states, waypoints=waypoints)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

def run_euler_f16_sim(init, tmax, step_size, waypoints):
    'run the simulation and return a list of states'

    autopilot = WaypointAutopilot(waypoints[0])

    cur_state = np.array(init)
    states = [cur_state.copy()] # state history

    cur_time = 0
    cur_waypoint_index = 0

    # waypoint distance and tolerance paramters
    wp_dist = 200
    wp_tol = 1e-6

    while cur_time + 1e-6 < tmax: # while time != tmax
        # update state
        cur_state = states[-1] + step_size * autopilot.der_func(cur_time, cur_state)

        states.append(cur_state)
        cur_time = cur_time + step_size

        # print distance to waypoint
        cur_waypoint = np.array(waypoints[cur_waypoint_index])

        x = cur_state[StateIndex.POS_E]
        y = cur_state[StateIndex.POS_N]
        z = cur_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)
        print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

    return states

def zero_crossing_function(cur_state, waypoint, wp_dist):
    x = cur_state[StateIndex.POS_E]
    y = cur_state[StateIndex.POS_N]
    z = cur_state[StateIndex.ALT]

    cur_pos = np.array([x, y, z])

    return np.linalg.norm(waypoint - cur_pos) - wp_dist

def bisection_zero_crossing(autopilot, waypoint, cur_time, last_state, step_size, wp_dist, wp_tol):
    a = 0
    b = step_size

    while abs(b - a) > wp_tol:
        mid = (a + b) / 2

        k1 = autopilot.der_func(cur_time - step_size, last_state)
        k2 = autopilot.der_func(cur_time - step_size, last_state + (mid / 2) * k1)
        k3 = autopilot.der_func(cur_time - step_size, last_state + (mid / 2) * k2)
        k4 = autopilot.der_func(cur_time - step_size, last_state + mid * k3)

        cur_state = last_state + (mid/6) * k1 + (mid/3) * k2 + (mid/3) * k3 + (mid/6) * k4

        if zero_crossing_function(cur_state, waypoint, wp_dist) > 0:
            a = mid
        else:
            b = mid

    return cur_state, mid

def rk45_bisection(autopilot, a_time, b_time, dense_function, waypoint, wp_dist, wp_tol):
    a = a_time
    b = b_time

    while abs(b - a) > wp_tol:
        mid = (a + b) / 2

        cur_state = dense_function(mid)

        if zero_crossing_function(cur_state, waypoint, wp_dist) > 0:
            a = mid
        else:
            b = mid

    return cur_state, mid

def run_rk4_classic_f16_sim(init, tmax, step_size, waypoints, log=True):
    'run the simulation using RK4 Classic Method Zero Crossing and return a list of states'

    print("Simulation based on RK4 Zero Crossing...")

    autopilot = WaypointAutopilot(waypoints[0])

    cur_state = np.array(init)
    states = [cur_state.copy()] # state history

    cur_time = 0
    cur_waypoint_index = 0
    zero_cross_flag = False

    # waypoint distance and tolerance paramters
    wp_dist = 200
    wp_tol = 1e-6

    while cur_time + 1e-6 < tmax: # while time != tmax
        # update state
        last_state = states[-1]
        k1 = autopilot.der_func(cur_time, last_state)
        k2 = autopilot.der_func(cur_time, last_state + (step_size / 2) * k1)
        k3 = autopilot.der_func(cur_time, last_state + (step_size / 2) * k2)
        k4 = autopilot.der_func(cur_time, last_state + step_size * k3)

        cur_state = last_state + (step_size/6) * k1 + (step_size/3) * k2 + (step_size/3) * k3 + (step_size/6) * k4 

        if zero_crossing_function(cur_state, waypoints[cur_waypoint_index], wp_dist) > 0:
            states.append(cur_state)
            cur_time = cur_time + step_size
        else:
            cur_state, zero_step_size = bisection_zero_crossing(autopilot, waypoints[cur_waypoint_index], cur_time, last_state, step_size, wp_dist, wp_tol)
            states.append(cur_state)
            cur_time = cur_time + zero_step_size
            print("************ State Changed! ************")
            zero_cross_flag = True


        # print distance to waypoint
        cur_waypoint = np.array(waypoints[cur_waypoint_index])

        x = cur_state[StateIndex.POS_E]
        y = cur_state[StateIndex.POS_N]
        z = cur_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)
        if zero_cross_flag == True or log == True:
            print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

        if zero_cross_flag == True:
            cur_waypoint_index += 1
            autopilot = WaypointAutopilot(waypoints[cur_waypoint_index])
            zero_cross_flag = False

    return states

def part3_simulations(init, tmax, step_size, waypoints):
    cur_step_size = step_size

    all_step_size = []
    all_final_states = []
    all_run_time = []
    all_estimate_errors = []

    while True:
        print(f"\n\n############# Simulating Using Step Size = {cur_step_size} #############\n\n")
        all_step_size.append(cur_step_size)

        start_time = time.perf_counter()

        states = run_rk4_classic_f16_sim(init, tmax, cur_step_size, waypoints, log=False)

        runtime = time.perf_counter() - start_time
        print(f"Simulation Completed in {round(runtime, 2)} seconds")
        all_run_time.append(runtime)

        # print final state
        final_state = states[-1]
        x = final_state[StateIndex.POS_E]
        y = final_state[StateIndex.POS_N]
        z = final_state[StateIndex.ALT]

        all_final_states.append(np.array([x, y, z]))
        if len(all_final_states) > 1:
            cur_estimate_error = np.linalg.norm(all_final_states[-2] - all_final_states[-1])
            all_estimate_errors.append(cur_estimate_error)

            if cur_estimate_error < 1:
                break
        cur_step_size /= 2

    print("############## Simulation Finished! ##############")
    print("All Step Size", all_step_size)
    print("All final_states", all_final_states)
    print("All run times", all_run_time)
    print("All estimate error", all_estimate_errors)

def run_rk45_f16_sim(init, tmax, step_size, waypoints, log=True):
    'run the simulation using RK45 Method Zero Crossing and return a list of states'

    print("Simulation based on RK45 Zero Crossing...")

    autopilot = WaypointAutopilot(waypoints[0])

    cur_state = np.array(init)
    states = [cur_state.copy()] # state history

    cur_time = 0
    last_time = 0

    cur_waypoint_index = 0
    zero_cross_flag = False

    atol = 1e-8
    rtol = 1e-5

    rk45 = RK45(autopilot.der_func, 0, init, tmax, rtol=rtol, atol=atol)

    # waypoint distance and tolerance paramters
    wp_dist = 200
    wp_tol = 1e-6 

    while cur_time + 1e-6 < tmax: # while time != tmax
        # update state 
        rk45.step()

        last_time = cur_time
        cur_time = rk45.t       
        cur_state = rk45.y

        if zero_crossing_function(cur_state, waypoints[cur_waypoint_index], wp_dist) > 0:
            states.append(cur_state)
        else:
            dense_function = rk45.dense_output()
            cur_state, zero_crossing_time = rk45_bisection(autopilot, last_time, cur_time, dense_function, waypoints[cur_waypoint_index], wp_dist, wp_tol)
            states.append(cur_state)
            cur_time = zero_crossing_time
            print("************ State Changed! ************")
            zero_cross_flag = True


        # print distance to waypoint
        cur_waypoint = np.array(waypoints[cur_waypoint_index])

        x = cur_state[StateIndex.POS_E]
        y = cur_state[StateIndex.POS_N]
        z = cur_state[StateIndex.ALT]

        cur_pos = np.array([x, y, z])
        distance = np.linalg.norm(cur_waypoint - cur_pos)
        if zero_cross_flag == True or log == True:
            print(f"Time {round(cur_time, 6)}, distance to waypoint: {round(distance, 3)} ft")

        if zero_cross_flag == True:
            cur_waypoint_index += 1
            autopilot = WaypointAutopilot(waypoints[cur_waypoint_index])
            rk45 = RK45(autopilot.der_func, cur_time, cur_state, tmax, rtol=rtol, atol=atol)
            zero_cross_flag = False

    return states    

if __name__ == '__main__':
    main()
