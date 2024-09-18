#!/usr/bin/env python3
#PBS -l nodes=bead56
#PBS -e error_harm_tosaap.e
#PBS -o output_harm_tosaap.o
import os, sys
if "PBS_O_WORKDIR" in os.environ:
    os.chdir(os.environ["PBS_O_WORKDIR"])
    sys.path.append(".")
sys.path.insert(0, "/net/debye/francescop/rumdpy-dev/")

'''
Code to obtain the Helmholtz free energy of the SAAP crystal.
Evaluate the mean (over the number of blocks of a NVT md simulation) of the 
difference between the saap potential and an harmonic potential of the same
atoms.
'''

import numpy as np
import rumdpy as rp
from time import time
from numba import cuda
from SAAP_potential import saap
import matplotlib.pyplot as plt
from Einstein_k import set_conf, set_NVTsim, k_Einstein_estimate

for i in range(3):
    try:
        cuda.select_device(i)
        break
    except Exception:
        pass

def evaluate_along_trajectory(sim, num_timeblocks, first_block, evaluater_list):
    mean_U = np.zeros(len(evaluater_list))
    U = np.zeros((len(evaluater_list), num_timeblocks - first_block))
    for block in sim.timeblocks():
        if block >= first_block:
            for i, ev in zip(range(len(evaluater_list)), evaluater_list):
                ev.evaluate(sim.configuration)
                U[i, block - first_block] = np.sum(ev.configuration['u'])
    mean_U = np.mean(U, axis=1)
    return mean_U

def get_mean_pressure(conf, sim, first_block):
    N = conf.N
    D = conf.D
    K, W, V = rp.extract_scalars(sim.output, ['K', 'W', 'Vol'], first_block=first_block)
    rho = N / V
    # Kinetic temperature
    dof = D * N - D # degrees of freedom
    T_kin = 2 * K / dof
    # Compute instantaneous pressure
    P = rho * T_kin + W / V
    print(f'mean pressure = {np.mean(P):.2f} +- {np.std(P):.2f}')
    return np.mean(P), np.std(P)

def save_hist(array, nbins, xlabel, ylabel, title, fname):
    mean = np.mean(array)
    std = np.std(array)
    plt.figure()
    plt.hist(array, bins=nbins)
    plt.axvline(mean, label=f'mean = {mean:.4f}', color='red')
    plt.axvline(mean + std, label=f'mean + std dev = {(mean+std):.4f}', color='orange')
    plt.axvline(mean - std, label=f'mean - std dev = {(mean-std):.4f}', color='orange')
    plt.xlabel(xlabel=xlabel, fontsize=15)
    plt.ylabel(ylabel=ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    plt.xticks()
    plt.yticks()
    plt.grid()
    plt.legend()
    plt.savefig(fname=fname)
    return

def plot_displacement(r, xlabel, ylabel, title, fname):
    r_mean = np.mean(r)
    time_v = np.linspace(0, len(r), num=len(r))
    plt.figure()
    plt.plot(time_v, r)
    plt.axhline(r_mean, label=f'mean = {r_mean:.3f}', color='red')
    plt.xlabel(xlabel=xlabel, fontsize=13)
    plt.ylabel(ylabel=ylabel, fontsize=13)
    plt.title(title, fontsize=15)
    plt.xticks()
    plt.yticks()
    plt.grid()
    plt.legend()
    plt.savefig(fname=fname)
    return

if __name__=='__main__':
    #### NUMBER OF INTEGRATION POINTS FOR EACH NVT ####
    n_int_points = 51
    lamb_v = np.linspace(0,1,n_int_points)

    #### SETTINGS OF SIMULATION AND CONFIGURATION OBJECTS ####
    # Define a reduced unit time scale
    m = 1.0
    T = 2.0 # temperature for which we want coexistence
    rho = 1.20
    t0 = rho**(1/3) / (T / m)**(1/2) # To work in reduced units of time (for dt and tau_T)

    # Simulation parameters
    D = 3
    nx, ny, nz = 8, 8, 8
    cell = [nx, ny, nz]
    N = 4 * nx * ny * nz # it's an FCC crystal
    dt = 0.001*t0
    num_timeblocks = 4096
    first_block = num_timeblocks // 8
    steps_per_timeblock = 64
    sim_params = [dt, num_timeblocks, steps_per_timeblock]
    # Coupling for the NVT Nose thermostat
    tau_T = 0.1*t0

    #### SAAP POTENTIAL ####
    # Setting starting parameters of the potential and cutoff
    sigma, eps = 1.0, 1.0
    pot_params = [65214.64725, -9.452343340, -19.42488828, 
                -1.958381959, -2.379111084, 1.051490962, 
                sigma, eps]
    cut = [4.0 * sigma]
    pot_params = pot_params + cut
    # Setup pair potential for evaluation
    pair_func_saap = rp.apply_shifted_potential_cutoff(saap) 
    pair_pot_saap_ev = rp.PairPotential(pair_func_saap, pot_params.copy(), max_num_nbs=1000)
    
    DeltaU_mean = np.zeros(lamb_v.shape)
    print(f"Density {rho:.2f}")
    # Simulation to estimate <(r-r0)^2>, to get k_Einstein
    configuration = set_conf(D, cell, rho, m, T)
    starting_positions = np.copy(configuration['r'])
    sim = set_NVTsim(configuration, num_timeblocks, steps_per_timeblock, pair_pot_saap_ev, T, tau_T, dt)
    tuning_factor = 1 ### The tuning is needed to avoid the divergence at the end of the integral
    k_Einstein, sim = k_Einstein_estimate(configuration, sim, starting_positions, first_block, T, tuning_factor=tuning_factor)[:2]
    print(f'current k_einstein = {k_Einstein}')
    # Simulation to get mean pressure for SAAP
    P_mean, P_std = get_mean_pressure(configuration, sim, first_block)
    for j in range(len(lamb_v)):
        print(f"Simulation {j+1} of {len(lamb_v)}")
        # Set configuration object
        configuration = set_conf(D, cell, rho, m, T)    
        # Set lambda saap potential
        pot_params[7] = (1 - lamb_v[j]) * eps
        pair_pot_saap_lambda = rp.PairPotential(pair_func_saap, pot_params, max_num_nbs=1000)
        # Set lambda harmonic potential
        harmonic_springs_lambda = rp.Tether()
        k_lambda = lamb_v[j] * k_Einstein
        harmonic_springs_lambda.set_anchor_points_from_types(particle_types=[0], spring_constants=[k_lambda], configuration=configuration)
        # Set simulation object with the mixed lambda potential
        sim = set_NVTsim(configuration, num_timeblocks, steps_per_timeblock,
                            [pair_pot_saap_lambda, harmonic_springs_lambda],
                            T, tau_T, dt)
        # Setup saap evaluator
        ev_saap = rp.Evaluater(configuration, pair_pot_saap_ev)
        # Setup harmonic evaluator
        lj = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
        none_interacting = rp.PairPotential(lj, params=[0.0, 1.0, 2.5], max_num_nbs=1000) # lj pair pot with sig = 0
        harmonic_springs_ev = rp.Tether()
        harmonic_springs_ev.set_anchor_points_from_types(particle_types=[0], spring_constants=[k_Einstein], configuration=configuration)
        ev_harmonic = rp.Evaluater(sim.configuration, [none_interacting, harmonic_springs_ev])
        # Evaluation of saap and harmonic potential along md trajectory
        t0 = time()
        U_saap, U_harm = evaluate_along_trajectory(sim, num_timeblocks, first_block, [ev_saap, ev_harmonic])
        print(f'time of the evaluation along trajectory: {(time()-t0):.1f}')
        DeltaU_mean[j] = U_saap - U_harm
        print(f'mean Delta U {j+1} = {DeltaU_mean[j]}, U_saap = {U_saap}, U_harm = {U_harm}')
    
    fname = f'data/rho={rho:.2f}'    
    tosave = np.array([DeltaU_mean, k_Einstein, P_mean, P_std, lamb_v], dtype=object)
    np.save(fname, tosave) 