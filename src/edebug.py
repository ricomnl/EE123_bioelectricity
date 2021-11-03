# Copyright 2018 Alexis Pietak and Joel Grodstein
# See "LICENSE" for further details.
    #import pdb; pdb.set_trace()
'''
Various functions to help with debug

dump(t, cc_cells, units, long):
    The main function for debug dumping.
    - In short mode ('long'=False), it just prints the time, cc_cells and Vm.
      In long mode, it prints a host more information.
    - 'Units' can be either mV_per_s, mol_per_m2s, or mol_per_m3s. Note that
      the moles_per_* fluxes are the amounts *entering* the cell. And, mV_per_s
      doesn't always treat negatively-charged molecules correctly.
    - By default, dump() prints most numbers as 3-digit integers in [0,999];
      you can get more precision by going into the function scale() and
      increasing N_DIGITS from 3 to, e.g., 5.

analyze_equiv_network (cc_cells, GP):
    This function prints a Thevenin equivalent circuit for the cell(s) at a
    particular point in time. I.e., cc_cells should be the N_IONSxN_CELLS
    array of ion concentrations at some time point. Note that if you're in a
    post_...() function, you can get the concentrations at the end of the
    simulation with cc_shots[-1].

    The first set of analyses is per cell; it models ICs and disregards GJs.
    It starts by analyzing each ion channel separately. For each of Na,K and Cl:
    - Compute and print Vnernst by the standard formula based on internal and
      external concentrations. 
    - Compute and print Gthev, using a linearized first-order model of drift
      current that works sort of OK (see  IC_Gthev() for details). This is
      trying to predict the amount that the ion's molar flux changes as Vmem
      departs from Vnernst.
      Gthev is negative for Na and K, and positive for Cl (again, see IC_Gthev()
      for an exact explanation of the Gthev sign).
    - print the total flux of the ion (using this imperfect model). This total
      flux is the approximate flux in moles/m2s; it does not include pumps. It
      is unlikely to be quite correct unless Vmem=Vnernst.

    Next, it merges the three equivalent circuits (Na, K and Cl) into one
    Thevenin equivalent that comprehends all three ions (but still no pumps).
    When doing so, it now comprehends that Cl is negative, and flips the Cl
    Gthev sign. As a result, this merged equivalent circuit predicts *charge*
    flux into the cell, not ion flux.

    The final per-cell computation is the open-circuit Vmem. It almost would
    simply print out the just-mentioned across-all-ions charge-based Vthev --
    except that this time it also includes the pump currents. Thus, it is the
    Vmem at which the merged equivalent circuit just above produces just enough
    charge flow to match the pumps. Note that this Vmem is based on the linear
    approximation above, and typically will not exactly predict the Vmem
    numbers as computed by GHK.

    Finally, if there are GJs, it prints more information on them (which I've
    not documented yet).

dump_gating ():
    Prints out the values of the gating for all ion channels and GJs.
    sim.cc_cells *must* already be set up.

There are also few utility routines:

Vnernst (p):
	Compute and return the Nernst voltage for Na, K and Cl for each cell's
	ion channels. Return an array [3,n_cell]
	Why only three rows? We only care about Na, K and Cl; they are the only
	ions that are charged and permeate the cell membrane. And neutral ions
	cause kT/q to be infinity. We assume that Na, K and Cl are
	cc_cells[0:3,:].

 IC_Gthev (p):
	The same idea as Vnernst. Units are moles/(m^2*s) per Volt.
	The main eqn is Gthev = (D/L) * Z * Cavg /26mV. It is a linear
	approximation to GHK, which works reasonably well.
	Note that it has ion-channel gating built in.
	Returns an array [3,n_cells] ("3" for Na, K and Cl).
'''
import numpy as np
import operator
import sim_toolbox as stb
import sim as sim

# The treatment of ion valence is arguably misleading:
# - When we print per-ion numbers, we are really always printing amounts of
#   moles, and disregarding whether valence is positive or negative. For things
#   like GJ drift currents, the particle flow numbers clearly take particle
#   charge into account -- but the numbers we print are in terms of moles (i.e.,
#   number of ions), not signed charge. This is still true even when we are
#   printing units of mV_per_s -- i.e., when we convert quantities from moles/s
#   to mV_per_s we essentially assume that every ion has valence=+1.
# - The final "grand valence-weighted totals" *are* based on actual charge, and
#   *do* take valence into account. When they are printed as mV_per_s, they are
#   thus correct. When they are printed as mol_per_m2s or mol_per_m3s, they
#   still take valence into account, and thus are arguably wrong. So why do we
#   do it that way? Because we often use these numbers to see if we're reached
#   QSS or not. 
def dump (t, cc_cells, units, long):
    print ('\nt={}: dumping {}-format...'.format(t,'long' if long else 'short'))

    # Print cc_cells[].
    sim.cc_cells = cc_cells
    print ('cc_cells= (moles/m3)')
    for ion_name,idx in sorted (sim.ion_i.items(),key=operator.itemgetter(1)):
        if (cc_cells[idx,:].sum()>0):
            fmt = '{:4.1f}' if (cc_cells[idx,:].max()<9.5) else '{:4.0f}'
            np.set_printoptions (formatter={'float': fmt.format})
            print ('   {} {}'.format(cc_cells[idx], ion_name))

    # Print Vm.
    np.set_printoptions (formatter={'float': '{:4.1f}'.format})
    print ('Vm =      {}mV'.format (sim.compute_Vm(cc_cells)*1000.0))
    if (not long):
        return

    (pump_Na, pump_K, GHK_fluxes, GJ_diff, GJ_drif, net_gen) \
      =sim_slopes_debug (t,cc_cells,sim.cc_env,sim.Dm_array,sim.z_array, \
                                  sim.ion_i,sim.GJ_connects,sim.GP)

    (scl,tag) = scale ([pump_Na,pump_K,GHK_fluxes,GJ_diff,GJ_drif,net_gen],
                       units,sim.GP)

    # We have the drift & diffusion fluxes by ion in each GJ; sum them to get
    # per-cell values.
    GJ_diff_by_cell = np.zeros (cc_cells.shape)
    GJ_drif_by_cell = np.zeros (cc_cells.shape)
    for ion in range(cc_cells.shape[0]):
        np.add.at(GJ_drif_by_cell[ion,:], sim.GJ_connects['from'],-GJ_drif[ion])
        np.add.at(GJ_drif_by_cell[ion,:], sim.GJ_connects['to'],   GJ_drif[ion])
        np.add.at(GJ_diff_by_cell[ion,:], sim.GJ_connects['from'],-GJ_diff[ion])
        np.add.at(GJ_diff_by_cell[ion,:], sim.GJ_connects['to'],   GJ_diff[ion])

    # For each ion, sum all the various contributions per cell.
    ion_totals = GHK_fluxes + GJ_diff_by_cell + GJ_drif_by_cell
    ion_totals[sim.ion_i['Na']] += pump_Na
    ion_totals[sim.ion_i['K']]  += pump_K

    # And also get the grand total flux (all ions together, for QSS) per cell
    # This is the *only* place we look at ion valences.
    grand_totals = (ion_totals.T * sim.z_array).T.sum(0)

    # Now the summing & distributing is done; convert to the proper units and
    # round to integers for printing.
    pump_Na = np.round(pump_Na*scl)
    pump_K  = np.round(pump_K*scl)
    GHK_fluxes = np.round(GHK_fluxes*scl)
    GJ_diff = np.round(GJ_diff*scl)
    GJ_drif = np.round(GJ_drif*scl)
    GJ_diff_by_cell = np.round(GJ_diff_by_cell*scl)
    GJ_drif_by_cell = np.round(GJ_drif_by_cell*scl)
    net_gen = np.round(net_gen*scl)
    ion_totals   = np.round (ion_totals*scl)
    grand_totals = np.round (grand_totals*scl)

    # First, ion channels and pumps
    np.set_printoptions (formatter={'float': '{:4.0f}'.format})
    for ion_name,ion_idx in sorted(sim.ion_i.items(),key=operator.itemgetter(1)):
        if ((GHK_fluxes[ion_idx,:]!=0).any()):
            print ('{:2s} ionCh: {} {}'.format (ion_name, GHK_fluxes[ion_idx], tag))
        if (ion_name=='Na'):
            print ('Na pump:  {} {}'.format(pump_Na, tag))
        if (ion_name=='K'):
            print ('K  pump:  {} {}'.format(pump_K, tag))

        if ((GJ_diff_by_cell[ion_idx,:]!=0).any()):
            print ('{:2s} GJdiff:{} {}'.format(ion_name,
                                                 GJ_diff_by_cell[ion_idx], tag))
        if ((GJ_drif_by_cell[ion_idx,:]!=0).any()):
            print ('{:2s} GJdrif:{} {}'.format(ion_name,
                                                 GJ_drif_by_cell[ion_idx], tag))
    # Per-ion totals
    print ('\nTotals by ion from all sources:')
    for ion_name,ion_idx in sorted(sim.ion_i.items(),key=operator.itemgetter(1)):
        if ((ion_totals[ion_idx,:]!=0).any()):
            print('{:2s} total: {} {}'.format(ion_name,ion_totals[ion_idx],tag))

    # Grand totals
    print ('\nGrand valence-weighted totals from all sources across all ions:')
    print ('          {} {}'.format(grand_totals, tag))

    # Now, GJs. We did them above by cell; this time, by GJ
    print ('\nnow the GJs by GJ (if any have non-zero flux):')
    for ion_name,ion_idx in sorted(sim.ion_i.items(),key=operator.itemgetter(1)):
        if ((GJ_diff[ion_idx,:]!=0).any()):
            print ('{:2s} GJ diff: {} {}'.format(ion_name,GJ_diff[ion_idx],tag))
        if ((GJ_drif[ion_idx,:]!=0).any()):
            print ('{:2s} GJ drif: {} {}'.format(ion_name,GJ_drif[ion_idx],tag))

    # Generation and decay
    if ((net_gen!=0).any()):
        print ('\nnow generation - decay:')
    for ion_name,ion_idx in sorted(sim.ion_i.items(),key=operator.itemgetter(1)):
        if ((net_gen[ion_idx,:]!=0).any()):
            print ('{:2s} net gen: {} {}'.format(ion_name,net_gen[ion_idx],tag))

def dump_gating ():
    print ('Gating summary:')
    np.set_printoptions (formatter={'float': '{:4.2f}'.format})
    for ion_index, ion in enumerate (['Na', 'K', 'Cl']):
        gates = sim.eval_gate (sim.IC_gates[ion_index,:])
        if ((gates==1.0).all()):
            print ('{:2} ionCh gating=[ none ]'.format (ion))
        else:
            print ('{:2} ionCh gating={}'.format (ion, gates))

    gates = sim.eval_gate (sim.GJ_gates)
    if ((gates==1.0).all()):
        print ('{:2} GJ gating=[ none ]'.format (ion))
    else:
        print ('   GJ gating={}'.format(gates))

# Print various Thevenin models. See above for details.
def analyze_equiv_network (cc_cells, GP):
    print ('Analyze_equiv_network() info...')

    # First, the ion-channel equivalents. And x1000 to use mV and not V.
    Vn = Vnernst(cc_cells,GP)*1000	# 3 rows(for Na,K,Cl) x num_cells
    Gthev = IC_Gthev(cc_cells,GP)/1000	# 3 rows x num_cells; moles/(m2*s)/mV

    # Now merge across ions; units are still V for Vthev; moles/(m2*s)/mV for G.
    (Vthev_net,Gthev_net) = merge_Thev(Vn,Gthev,sim.z_array)

    # Actual fluxes at current Vm
    Vm = sim.compute_Vm (cc_cells)*1000
    IC_flux = (Vm-Vn)*Gthev		#[3,num_cells] of IC flux; moles/(m2*s)
    IC_flux_net=IC_flux[0]+IC_flux[1]-IC_flux[2] # summed across all ions

    # Make the Gthev numbers pretty for printout. Their current units are
    # moles/(m2*s)/mV and we convert them to moles/(m3*s)/mV, which is more
    # commonly used for the ion pumps.
    # Scale() always assumes it's getting moles/m2s and converts that to
    # moles/m3s for us; the extra /mV on the input & output doesn't change the
    # scaling (but we'll eventually have to print "/mV" manually!)
    (scl,tag) = scale ([Gthev, Gthev_net], Units.mol_per_m3s, GP)
    Gthev     = np.round (Gthev*scl)
    Gthev_net = np.round (Gthev_net*scl)
    # When we scale Gthev, we must also scale fluxes correspondingly.
    IC_flux = np.round (IC_flux*scl)		# 3 rows x num_cells
    IC_flux_net = np.round (IC_flux_net*scl)	# [num_cells]

    # Print out Vn, Gth for each ion channel.
    for ion_index,ion in enumerate (['Na', 'K', 'Cl']):
        np.set_printoptions (formatter={'float': '{:4.2g}'.format})
        print ('    {:2s} IC Vn ={}mV'.format (ion, Vn[ion_index]))
        np.set_printoptions (formatter={'float': '{:4.0f}'.format})
        print ('    {:2s} IC Gth={}{}'.format (ion, Gthev[ion_index],tag+'/mV'))
        print ('    {:2s} IC curr={}{}{}'.format (ion, IC_flux[ion_index],tag,
               ' of ions through ICs'))

    # Print out Vn, Gth for the merged-across-all-ion-channels version.
    np.set_printoptions (formatter={'float': '{:4.2g}'.format})
    print ('IC Vthev_net ={}mV'.format (Vthev_net))
    np.set_printoptions (formatter={'float': '{:4.0f}'.format})
    print ('IC Gth_net   ={}{}'.format (Gthev_net,tag+'/mV'))
    print ('IC curr_net   ={}{}{}'.format (IC_flux_net, tag,
           ' of charge through ICs'))

    # Next, equivalents for GJs (if there are any).
    deltaV_GJ = (Vm[sim.GJ_connects['to']]-Vm[sim.GJ_connects['from']])
    (GJ_Ith, GJ_Gth) = sim.GJ_norton (deltaV_GJ)
    if (GJ_Gth.size == 0):
        return

    # Scale the GJ models for printing.
    units = Units.mol_per_m2s
    units = Units.mV_per_s
    (scl,tag) = scale ([GJ_Gth, GJ_Ith],units, GP)
    GJ_Gth = np.round(GJ_Gth*scl)
    GJ_Ith = np.round(GJ_Ith*scl)

    # Print the per-ion GJ models.
    for ion, ion_index in sorted (sim.ion_i.items(),key=operator.itemgetter(1)):
        if ((GJ_Ith[ion_index,:]!=0).any()):
            print ('    {:2} GJ Ith   =  {}{}'.format(ion,GJ_Ith[ion_index],tag))
        if ((GJ_Gth[ion_index,:]!=0).any()):
            print ('    {:2} GJ Gth   =  {}{}'.format(ion,GJ_Gth[ion_index],tag+'/V'))

###################################################################
# Utility functions
###################################################################

# Mostly like sim.sim_slopes(). However, this version
# - returns all of the individual data that we can coalesce however we like
# - returns it all in units of moles/(m2*s), not moles/(m3*s). And it's the
#   molar flux *into* the cells.
# - the GJ data is all mass fluxes in the from->to direction.
def sim_slopes_debug (t,Cc_cells,cc_env,Dm_array,z_array,ion_i,GJ_connects,GP):
    sim.cc_cells = Cc_cells
    Vm = sim.compute_Vm (Cc_cells)

    # General note: our units of flux are moles/(m2*s). The question: m2 of
    # what area? You might think that for, e.g., ion channels, it should be per
    # m2 of ion-channel area -- but it's not. All fluxes are per m2 of cell-
    # membrane area. Thus, the (e.g.,) diffusion rate through ion channels must
    # be scaled down by the fraction of membrane area occupied by channels.
    # The same goes for ion pumps and GJs.

    # Run the Na/K-ATPase ion pump in each cell.
    # Returns two 1D arrays[N_CELLS] of fluxes; units are moles/(m2*s)
    pump_Na,pump_K,_ = stb.pumpNaKATP(Cc_cells[ion_i['Na']],cc_env[ion_i['Na']],
                                      Cc_cells[ion_i['K']], cc_env[ion_i['K']],
                                      Vm, GP.T, GP, 1.0)

    # Kill the pumps on worm-interior cells (based on Dm=0 for all ions)
    keep_pumps = np.any (Dm_array>0, 0)	# array[n_cells]
    pump_Na *= keep_pumps
    pump_K  *= keep_pumps

    # GHK on the ion channels.
    f_GHK = stb.GHK (cc_env, sim.cc_cells, Dm_array, z_array, Vm, GP)
    for g in sim.IC_gates:
        f_GHK[g.out_ion] *= g.func (g, Cc_cells, Vm, t)
    GHK_fluxes = f_GHK

    # Gap-junction computations.
    deltaV_GJ = (Vm[GJ_connects['to']] - Vm[GJ_connects['from']]) # [n_GJs]

    # Get the gap-junction Norton-equivalent circuits for all ions at once.
    # Units of Ith are mol/(m2*s); units of Gth are mol/(m2*s) per Volt.
    (GJ_Ith, GJ_Gth) = sim.GJ_norton(deltaV_GJ)	# Both are [n_ions,n_GJs].

    # As opposed so sim.slew_cc(), our callers just want GJ diffusion and
    # drift.
    # [n_ions,n_GJs] * [n_GJs]
    GJ_diff = GJ_Ith.copy()		# [n_ions,n_GJs]
    GJ_drif = GJ_Gth * deltaV_GJ	# [n_ions,n_GJs]
    # [n_ions,n_GJs] * [n_GJs] * [n_GJs]
    for g in sim.GJ_gates:
        scale = g.func (g, Cc_cells, Vm, t)
        GJ_diff *= scale
        GJ_drif *= scale

    # Next, do net generation. These are already natively in moles/(m3*s).
    # All arrays are [n_ions, n_cells]
    net_gen = np.zeros (Cc_cells.shape)
    for g in sim.GD_gates:
        net_gen[g.out_ion] += g.func (g, Cc_cells, Vm, t)

    # But unlike sim_slopes(), we return it in moles/(m2*s) instead.
    # So the usual conversion from moles/(m2*s) to moles/(m3*s) -- but reversed.
    net_gen *= (GP.cell_vol / GP.cell_sa)

    # All returned values are in moles/(m2*s), where the m2 is m2 of
    # cell-membrane area.
    return (pump_Na, pump_K, GHK_fluxes, GJ_diff, GJ_drif, net_gen)

from enum import Enum
class Units(Enum):
    mV_per_s    = 1
    mol_per_m2s = 2
    mol_per_m3s = 3

# scale (arrays, units, GP)
#   Scale arrays for nice printing.
#   Inputs:
#	'Arrays' is a tuple of numpy arrays; each has units of moles/(m2*sec).
#	'Units' is one of the Unit enums; it is our destination units.
#   Operations:
#     -	The input 'arrays' are all assumed to have units of moles/(m2*sec).
#	First compute the scale 'factor' to simply convert from moles/(m2*sec)
#	to 'units'. This must be applied simply for unit correctness.
#     -	Next, ease the formatting: pick a power-of-10 scale factor (an integer
#	x10) so as to be able to print the scaled numbers with integers in
#	[-999,999]. Specifically, find x10 such that, for all numbers 'n' in any
#	the arrays, round(abs(n*factor*(10^x10))) is <1000.
#   Return:
#      - the final scale factor (i.e., factor * 10^x10). Note that it includes
#        anything needed to transform units.
#      - 'tag' (a printable string representing the desired units). E.g., if
#	 we want mV/s, and x10 was 5, then 'tag' would be 'mV/s*10^5'.
#   Issues:
#     -	If our output units are mV_per_s, then we always just assume valence=1
#	(and we don't know the valence, anyway).
#     - We assume our input units to be moles/(m2*sec). Sometimes they're not;
#	e.g., analyze_equiv_network() gives us Gth as moles/(m2*sec) per Volt
#	of Vm. In that case, the caller must modify the 'tag' we return.
def scale (arrays, units, GP):
    import math
    N_DIGITS=3		# Print integers in [0,999]
    if (units == Units.mV_per_s):
        # mole/(m2*s) * (C/mole) / (F/m2) = V/s
        factor = 1000.0 * GP.F / GP.cm
        tag = "mV/s"
    elif (units == Units.mol_per_m2s):
        factor = 1.0
        tag = "mol/m2s"
    elif (units == Units.mol_per_m3s):
        factor = (GP.cell_sa / GP.cell_vol)
        tag = "mol/m3s"

    M=-100
    for ar in arrays:
        if (ar.size > 0):	# Max() crashes on zero-size arrays :-(
            M = max(M, np.abs(ar).max())
    assert M>=0, "You are trying to dump out NaN or similar illegal values"

    # Pick x10 such that (M*factor)*(10^x10) is in [100,999]
    x10 = 0 if (M==0) else math.floor (math.log10(1/(M*factor))) + N_DIGITS
    if (x10 != 0):
        tag = tag + '*10^' + str(x10)

    return (factor * pow(10,x10), tag)

# Given a scalar Vm, ion index and cell index run GHK that cell for that
# (Vm, ion_index).
# Return & print the results.
def GHK_debug (V, ion_name, cell, p):
    ion_index = sim.ion_i[ion_name]
    num_cells = sim.cc_cells.shape[1]
    Vm = np.zeros(num_cells); Vm[cell]=V
    f = stb.GHK (cc_env[ion_index], cc_cells[ion_index],
                 Dm_array[ion_index], z_array[ion_index], Vm, GP)
    #print ('V={}mV => {} flux={}'.format(V*1000, ion_name, f[cell]))
    return (f[cell])

# THIS FUNCTION IS FOR DEBUG ONLY (it's not used in simulation at all).
# In fact, it's only used by analyze_equiv_network()
# Compute and return the Nernst voltage for Na, K and Cl for each cell.
# Return an array [3,n_cell]
# Why only three rows? We only care about Na, K and Cl; they are the only ions
# that are charged and permeate the cell membrane. And neutral ions cause kT/q
# to be infinity. We assume that Na, K and Cl are cc_cells[0:3,:].
def Vnernst (cc_cells, p):
    # Nernst voltage is -(kT/q) * ln(Cint / Cext), where q is the charge of the
    # ion. Then, since q = Z*q_e, we have Vnernst = -(kT/q_e)/Z * ln(Cint/Cext)
    # = 26mV * ln(Cext/Cint)/Z
    k26mV = p.R * p.T / p.F	# a.k.a. kT/q
    #Vn[i,c] = .026 * np.log(cc_env[i] / cc_cells[i,c]) / z_array[i]
    Vn = (k26mV * np.log(sim.cc_env[0:3] / cc_cells[0:3,:].T) / sim.z_array[0:3]).T
    return (Vn)

    # Gthev = (D/L) * Z * Cavg /26mV 		# moles/(m^2*sec) per Volt
    # Gthev[i,c] = D[i,c]*z_array[i]*(.45*cc_cells[i,c]+.55*cc_env[i])/ (L*26mV)

# THIS FUNCTION IS FOR DEBUG ONLY (it's not used in simulation at all).
# In fact, it's only used by analyze_equiv_network()
#
# The same idea as Vnernst.
# Returns an array [3,n_cells] ("3" for Na, K and Cl).
# It is currently used only by analyze_equiv_network().
# The units on Gthev are (ions entering the cell/m2s) per Volt of Vmem; thus it
# tends to be negative for Na and K, and positive for Cl.
# Roughly, if the cell were a Norton equivalent circuit, the Ith would represent
# diffusion (which is independent of Vmem) and the Gth would represent drift.
# Note that the returned IC_Gthev has the scaling for IC gating built in.
#
# The main eqn is Gthev = -(D/L) * Z * conc_avg /26mV.
# It is not really based on GHK, but rather is just the standard drift equation:
#	flux=V*G, so
#	G = flux/V = conc * veloc / V = conc * Z * E * mobility / V
#	  = -conc * Z * mobility / L = -conc * Z * (D/26mV) / L as desired.
# The only hard part is deciding what to use for the concentration; we get the
# best fit to GHK with conc = .6 * conc_internal + .4 * conc_env.
def IC_Gthev (cc_cells, p):
    k26mV = p.R * p.T / p.F	# a.k.a. kT/q

    # Gthev[i,c] = D[i,c]*z_array[i]*(.45*cc_cells[i,c]+.55*cc_env[i])/ (L*26mV)
    cc_avg = (.6*cc_cells[0:3,:].T+.4*sim.cc_env[0:3]).T
    #cc_avg = np.minimum (cc_cells[0:3,:].T,sim.cc_env[0:3]).T
    Gthev = -sim.Dm_array[0:3,:].T * sim.z_array[0:3] * cc_avg.T / (p.tm*k26mV)
    #Gthev = Gthev * z_array[0:3] * p.F * p.cell_sa	# coul/sec per Volt
    Gthev = Gthev.T

    # Build [n_ions,n_cells] array of how much to gate (i.e., scale) Dm values.
    gate = np.ones_like (cc_cells)
    for g in sim.IC_gates:
        gate[g.out_ion,:] *= g.func (g, cc_cells, sim.Vm, 0)

    Gthev[0,:] *= gate[0]
    Gthev[1,:] *= gate[1]
    Gthev[2,:] *= gate[2]

    return (Gthev)

# merge_Thev (Vthev, Gthev, z_array):
#   THIS FUNCTION IS FOR DEBUG ONLY (it's not used in simulation at all).
#   In fact, it's only used by analyze_equiv_network().
#
#   Merge the Thevenin equivalent circuits for multiple ions into one net model.
#   Specifically: takes two arrays: Vthev(n_ions,n_items) and Gthev(ditto).
#   Merges them into a single Vthev(n_items) and Gthev(n_items).
#   'N_items' would typically be n_cells; we merge across ions, and each cell is
#   analyzed completely separately from each other cell. (We could use this same
#   function to create a net equivalent circuit for a GJ as well, if we liked).
#
#   The merge is a bit weird, due to the units. The incoming Vthev is V, and
#   Gthev is mol/(m2*s) into the cell per Volt, which usually results in
#   positively-charged species having negative Gthev. This is correct; higher
#   Vmem results in less Na and K coming in, and more Cl. However, we really
#   want to merge so that the resultant *charge* flux works, not the particle
#   flux. Thus, really all species (including Cl) should have negative Gth.
#   So we must scale each incoming Gthev by its ion's valence. So, +1
#   ions work normally, 0 ions must be ignored, etc. 
# The basic equations:
#	Geq = sum over all ions i of (Gthev_i * z_i)
#	Veq = (sum over all ions i of (Vth_i * Gthev_i)) / Geq
def merge_Thev (Vthev, Gthev, z_array):
    # For each cell, net Gthev is just of sum over each ion.
    Gthev = (Gthev.T * z_array[0:3]).T	# Scale by valence, as described
    Gthev_net = Gthev.sum(axis=0)	# So Gthev_net[3], one element per ion.

    # V_net = (V1*G1 + V2*G2 + V3*G3)/(G1+G2+G3)
    Vthev_net = Vthev * Gthev	# still [n_ions,n_items]
    Vthev_net = Vthev_net.sum(axis=0) / Gthev_net
    return (Vthev_net, Gthev_net)