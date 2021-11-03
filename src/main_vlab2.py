#!/usr/bin/env python3

# Copyright 2018 Alexis Pietak and Joel Grodstein
# See "LICENSE" for further details.

#    import pdb; pdb.set_trace()
import re
import sys

import numpy as np
import sim as sim
import eplot as eplt
import edebug as edb

#####################################################
# For an overview of the network data structures, see sim.py

# This is the function we're using for lab #1
def setup_lab2(p):
    p.adaptive_timestep = False	# Slow but accurate mode
    p.sim_dump_interval = 10	# Not much debug printout during the sim
    p.sim_long_dump_interval = 10

    num_cells = 4
    n_GJs = 0
    sim.init_big_arrays (num_cells, n_GJs, p)

    # By default, we have D_Na = D_K = 1e-18
    Na = sim.ion_i['Na']; K = sim.ion_i['K']
    sim.Dm_array[Na, 0] = 20.0e-18	# +44 mV (reversal potential of Na+)
    sim.Dm_array[Na, 1] = 5.0e-18	# +13mV
    sim.Dm_array[K, 2] = 10.0e-18	# -57mV
    sim.Dm_array[K, 3] = 160.0e-18	# -82mV (reversal potential of K+)

def post_lab2(GP, t_shots, cc_shots):
    eplt.plot_Vmem (t_shots, cc_shots)
    eplt.plot_ion(t_shots, cc_shots, "Na")

# This function shows that the QSS Vmem is insensitive to initial Vmem.
# 4 cells. Each one has Dm_Na set so that QSS Vmem will be -57mV.
# Tweak initial [Na] in cells 1-3 so that each starts out 5mV higher than
# the previous one. Nonetheless, they all converge to the same 30mV very
# quickly.
def setup_lab2b(p):
    p.adaptive_timestep = False	# Slow but accurate mode
    p.sim_dump_interval = 1000	# Not much debug printout during the sim

    num_cells = 4

    n_GJs = 0
    sim.init_big_arrays(num_cells, n_GJs, p)

    # By default, we have D_Na = D_K = 1e-18
    Na = sim.ion_i['Na']
    K = sim.ion_i['K']
    sim.Dm_array[K, :] = 10.0e-18	# -57mV

    sim.cc_cells[Na, 0] -= 0.025
    sim.cc_cells[Na, 1] -= 0.020
    sim.cc_cells[Na, 2] -= 0.015
    sim.cc_cells[Na, 3] -= 0.010
    
def post_lab2b(GP, t_shots, cc_shots):
    eplt.plot_Vmem(t_shots, cc_shots)
    eplt.plot_ion(t_shots, cc_shots, "Na")

######################################################################
######################################################################

def command_line ():
    # If no command-line arguments are given, prompt for them.
    if (len(sys.argv) <= 1):
        args = 'main.py ' + input('Arguments: ')
        sys.argv = re.split (' ', args)

    if (len(sys.argv) != 3):
       raise SyntaxError('Usage: python3 main.py test-name-to-run sim-end-time')

    end_time = float(sys.argv[2])
    name = sys.argv[1]

    # Run whatever test got chosen on the command line.
    GP = sim.Params()
    eval('setup_'+name+'(GP)')

    # Initialize Vmem -- typically 0, or close to that.
    Vm = sim.compute_Vm(sim.cc_cells)
    assert(np.abs(Vm)<.5).all(), \
            "Your initial voltages are too far from charge neutral"
    np.set_printoptions(formatter={'float': '{: 6.3g}'.format}, linewidth=90)
    print('Initial Vm   ={}mV\n'.format(1000*Vm))

    print('Starting main simulation loop')
    t_shots, cc_shots = sim.sim(end_time)
    print('Simulation is finished.')

    # We often want a printed dump of the final simulation results.
    np.set_printoptions(formatter={'float': '{:.6g}'.format}, linewidth=90)
    edb.dump(end_time, sim.cc_cells, edb.Units.mol_per_m3s, True)
    edb.analyze_equiv_network(sim.cc_cells, GP)
    #edb.dump (end_time, sim.cc_cells, edb.Units.mV_per_s, True)

    # Run a post-simulating hook for plotting, if the user has made one.
    if ('post_'+name in dir(sys.modules[__name__])):
        eval('post_'+name+'(GP, t_shots, cc_shots)')

command_line()