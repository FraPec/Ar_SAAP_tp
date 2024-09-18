#!/usr/bin/env python3
#PBS -l nodes=bead61
#PBS -e error_el_const.e
#PBS -o error_el_const.o
import os, sys
if "PBS_O_WORKDIR" in os.environ:
    os.chdir(os.environ["PBS_O_WORKDIR"])
    sys.path.append(".")
sys.path.insert(0, "/net/debye/francescop/rumdpy-dev/")

import numpy as np
import matplotlib.pyplot as plt
import rumdpy as rp
from numba import cuda, njit
import numba
from SAAP_potential import saap

for i in range(3):
    try:
        cuda.select_device(i)
        break
    except Exception:
        pass

@njit
def dist_sq_function(ri, rj, sim_box):  
    dist_sq = numba.float32(0.0)
    D = len(sim_box)
    for k in range(D):
        dr_k = ri[k] - rj[k]
        box_k = sim_box[k]
        dr_k += (-box_k if numba.float32(2.0) * dr_k > +box_k else
                    (+box_k if numba.float32(2.0) * dr_k < -box_k else numba.float32(0.0)))  # MIC
        dist_sq = dist_sq + dr_k * dr_k
    return dist_sq


def set_conf(D, cells, rho, m, T_init):
    #### CONFIGURATION OBJECT ####
    configuration = rp.Configuration(D=D)
    configuration.make_lattice(rp.unit_cells.FCC, cells=cells, rho=rho)
    configuration['m'] = m
    configuration.randomize_velocities(T=T_init)
    return configuration

def set_NVTsim(configuration, num_timeblocks, steps_per_timeblock, potentials, T, tau_T, dt):
    # Setup integrator: NVT
    integrator = rp.integrators.NVT(temperature=T, tau=tau_T, dt=dt)

    # Setup Simulation. Total number of time steps: num_blocks * steps_per_block
    sim = rp.Simulation(configuration, potentials, integrator,
                        num_timeblocks=num_timeblocks,
                        steps_per_timeblock=steps_per_timeblock,
                        steps_between_momentum_reset=0, 
                        storage='memory')
    return sim

def k_Einstein_estimate(conf, sim, starting_positions, first_block, T, tuning_factor):
    # Function to esteem the k of an Einstein crystal from a simulation
    potential = sim.interactions
    evaluater = rp.Evaluater(conf, potential)
    print('Estimate of Einstein crystal el. constant')
    sim_box = conf.simbox.lengths
    dr2_particle_v = np.zeros((conf.N, sim.num_blocks))
    evaluater.evaluate(sim.configuration)
    for block in sim.timeblocks():
        for particle in range(conf.N):
            dr2_particle_v[particle, block] = dist_sq_function(starting_positions[particle,:], conf['r'][particle, :], sim_box)
    dr2_particles = np.mean(dr2_particle_v[:,first_block:], axis=1)
    dr2_tot = np.sum(dr2_particle_v[:,first_block:], axis=0)
    k_Einstein_v = 3 * T / dr2_particles * tuning_factor # elastic constant vector, one constant per particle
    k_Einstein = np.mean(k_Einstein_v) * tuning_factor
    return k_Einstein, sim, dr2_particles, k_Einstein_v, dr2_tot

if __name__=='__main__':
    #### SETTINGS OF SIMULATION AND CONFIGURATION OBJECTS ####
    # Define a reduced unit time scale
    m = 1.0
    T = 2.0 # temperature for which we want coexistence
    rho = 1.2
    t0 = rho**(1/3) / (T / m)**(1/2) # To work in reduced units of time (for dt and tau_T)
    # Simulation parameters
    D = 3
    nx, ny, nz = 8, 8, 8
    cell = [nx, ny, nz]
    N = 4 * nx * ny * nz # it's an FCC crystal
    dt = 0.001*t0
    num_timeblocks = 2048
    first_block = num_timeblocks // 8
    steps_per_timeblock = 32
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

    configuration = set_conf(D, cell, rho, m, T)
    starting_positions = np.copy(configuration['r'])
    # Setup integrator: NVT
    integrator = rp.integrators.NVT(temperature=T, tau=tau_T, dt=dt)

    # Setup Simulation. Total number of time steps: num_blocks * steps_per_block
    sim = rp.Simulation(configuration, pair_pot_saap_ev, integrator,
                        num_timeblocks=num_timeblocks,
                        steps_per_timeblock=steps_per_timeblock,
                        steps_between_momentum_reset=100, 
                        storage='memory')


    tuning_factor=1


    k_Einstein, sim, dr2_saap, k_Einstein_v, dr2_tot_saap = k_Einstein_estimate(configuration, sim, starting_positions, first_block, T, tuning_factor=tuning_factor)
    print(f'Current k = {k_Einstein}')

    # TEST THE OBTAINED ELASTIC CONSTANT
    configuration = set_conf(D, cell, rho, m, T)
    starting_positions = np.copy(configuration['r'])
    harmonic_springs = rp.Tether()  #  U = 0.5*k*(r-r0)^2
    harmonic_springs.set_anchor_points_from_types(particle_types=[0],
                                                  spring_constants=[k_Einstein],
                                                  configuration=configuration 
                                                  )
    #harmonic_springs.set_anchor_points_from_lists(particle_indices=list(range(N)),
    #                                               spring_constants=[k_Einstein]*np.ones(N),
    #                                               configuration=configuration
    #                                               )

    lj = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    none_interacting = rp.PairPotential(lj, params=[0.0, 1.0, 2.5], max_num_nbs=1000) # lj pair pot with sig = 0
    sim = set_NVTsim(configuration, num_timeblocks, steps_per_timeblock, [none_interacting, harmonic_springs], T, tau_T, dt)
    sim_box = configuration.simbox.lengths
    Dimensions = configuration.D

    dr2_particle_v = np.zeros((N, num_timeblocks))
    for block in sim.timeblocks():
        for particle in range(N):
            dr2_particle_v[particle, block] = dist_sq_function(starting_positions[particle,:], configuration['r'][particle, :], sim_box)
    dr2_harm = np.mean(dr2_particle_v[:,first_block:], axis=1)
    dr2_tot_harm = np.sum(dr2_particle_v[:,first_block:], axis=0)

    print(f'mean r^2 saap = {np.mean(dr2_saap)}, mean r^2 harm (all equal k) = {np.mean(dr2_harm)}')
    print(dr2_saap)
    print(dr2_harm)


    # Plot histograms to check

    save = 0
    show = 1

    fig0 = plt.figure(0)
    plt.title(r'Hist. of $\langle (r_i-r_{i,0})^2 \rangle$_{MD}')
    plt.hist(dr2_saap, bins=20, label='mean over MD blocks of dr2_saap', alpha=0.5)
    plt.hist(dr2_harm, bins=20, label='mean over MD blocks of dr2_harm', alpha=0.5)
    plt.grid()
    plt.legend()

    fig1 = plt.figure(1)
    plt.title(r'Hist. of occurrences of $\sum_i (r_i-r_{i,0})^2$ along MD simulation')
    plt.hist(dr2_tot_saap, bins=20, label=f'sum over {N} particles for saap', alpha=0.5)
    plt.hist(dr2_tot_harm, bins=20, label=f'sum over {N} particles for harm', alpha=0.5)
    plt.grid()
    plt.legend()

    fig2 = plt.figure(2)
    plt.hist(k_Einstein_v, bins=30, label='k_Einstein', alpha=0.5)
    plt.axvline(np.mean(k_Einstein), label=f'mean = {np.mean(k_Einstein):.1f}')
    plt.grid()
    plt.legend()

    if show==1:
        plt.show()
    if save==1:
        fig0.savefig(f'dr2_hist_tuning{tuning_factor}.pdf')
        fig1.savefig(f'dr2_tot_hist_tuning{tuning_factor}.pdf')
        fig2.savefig(f'k_hist_tuning{tuning_factor}.pdf')

