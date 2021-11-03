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
def setup_lab1(p):
    p.adaptive_timestep = True	# Fast (but sometimes unstable) mode
    p.sim_dump_interval = 1000	# Not much debug printout during the sim

    num_cells = 4		# 4 cells

    # Gap junctions connect cells (we'll learn more soon).
    # No gap junctions means that the 4 cells are independent.
    n_GJs = 0

    # We've declared how many cells and GJs the simulation has. Create them
    # all, with default values.
    sim.init_big_arrays(num_cells, n_GJs, p)

    # To make it easier to set per-ion values below.
    Na = sim.ion_i['Na']
    K = sim.ion_i['K']
    Cl = sim.ion_i['Cl']

    # We want this lab to use a little different default values than the usual
    # ones. Specifically, a different diffusion constant for the K ion channels.
    # Note that this is the baseline value; simulation #3 will double it.
    sim.Dm_array[K, :] = 10.0e-18

    # Simulation #2 asks you to change the initial cell-internal concentrations.
    # Here's where you can do that.
    # The defaults (set in sim.init_big_arrays) are [Na+]=10, [K+]=125,
    # [Cl-]=55, [other-]=80 (units are all moles/m^3). Note that they add up to
    # charge neutral. Make sure to keep the cell interior pretty close to charge
    # neutral, or else the simulation will explode!
    # For example, if we want to double [Na+], it would go from 10 to 20. We
    # will also increase [Cl] by 10 moles/m^3 to preserve charge neutrality.
    # You should do similar things for the other cells, as per the instructions.
    # Cell #1
    sim.cc_cells[Na, 1] += 10	# Cell #1: [Na] is 20 rather than 10
    sim.cc_cells[Cl, 1] += 10	# [Cl] changes to keep charge neutrality.
    # Cell #2
    sim.cc_cells[K, 2] += 125
    sim.cc_cells[Cl, 2] += 125
    # Cell #3
    sim.cc_cells[Cl, 3] += 55
    sim.cc_cells[K, 3] += 55

    # Simulation #3 asks you to alter the diffusion constants D_Na, D_K and
    # D_Cl. By default, sim.init_big_arrays() sets all three to 1e-18, and then
    # we overrode D_K above to 10e-18. Here's where we override them for
    # simulation #3.
    # Cell #1 doubles D_Na
    sim.Dm_array[Na, 1] += sim.Dm_array[Na, 1]
    # Cell #2 doubles D_K
    sim.Dm_array[K, 2] += sim.Dm_array[K, 2]
    # Cell #3 doubles D_Cl
    sim.Dm_array[Cl, 3] += sim.Dm_array[Cl, 3]

def post_lab1(GP, t_shots, cc_shots):
    eplt.plot_Vmem(t_shots, cc_shots)
    eplt.plot_ion(t_shots, cc_shots, "Na")
    eplt.plot_ion(t_shots, cc_shots, "K")
    eplt.plot_ion(t_shots, cc_shots, "Cl")


######################################################################
######################################################################

def command_line ():
    # If no command-line arguments are given, prompt for them.
    if (len (sys.argv) <= 1):
        args = 'main.py ' + input('Arguments: ')
        sys.argv = re.split (' ', args)

    if (len(sys.argv) != 3):
        raise SyntaxError('Usage: python3 main.py test-name-to-run sim-end-time')

    end_time = float (sys.argv[2])
    name = sys.argv[1]

    # Run whatever test got chosen on the command line.
    GP = sim.Params()
    eval ('setup_'+name+'(GP)')

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
    #edb.dump (end_time, sim.cc_cells, edb.Units.mV_per_s, True)

    # Run a post-simulating hook for plotting, if the user has made one.
    if ('post_'+name in dir(sys.modules[__name__])):
        eval('post_'+name+'(GP, t_shots, cc_shots)')

if __name__ == "__main__":
    command_line()