# Copyright 2018 Alexis Pietak and Joel Grodstein
# See "LICENSE" for further details.

'''
Big picture: what does the network look like?
    The network is just a bunch of cells and GJs. However, there is no "class
    cell" or "class GJ".
    Instead, our data structure is most just a bunch of arrays.
    Some (like the membrane voltage, Vm) are 1D arrays[N_CELLS], with 1 element
    per cell. Others (like the ion concentration, cc_cells) are 2D
    arrays[N_IONS,N_CELLS], and can thus track (e.g.,) the concentration of
    every ion in every cell independently. By keeping the information in arrays,
    we get to use canned matrix-math routines in numpy for efficient simulation.

Which arrays track the cells?
    The first set of arrays are all [N_IONS,N_CELLS]; they have one row per ion,
    and one column per cell.
	cc_cells[i,c] = current concentration of ion #i in cell #c. It is given
	  in moles/m3, which is equivalent to mmoles/liter.
	Dm_array[i,c] = membrane ion-channel diffusion constant of cell #c for
	  ion #i (m2/s). One might think that Dm would just be the diffusion
	  constant of an ion channel itself, and would remain constant and be
	  mostly identical across all cells. Instead, Dm merges this with the
	  fraction of the cell's surface area that is covered by channels. So
	  if 1/3 of a cell membrane were covered by ion channels, then we would
	  cut that cell's Dm down by 3x. Also note that most ion channels use
          facilitated diffusion, and so these diffusion constants may be
          unrelated to an ion's "normal" diffusion constant in water.
    The next set are all arrays[N_CELLS]:
	z_array[i] = valence of ion #i. This array remains constant.
	cc_env [i] = concentration of ion #i in the ECF (moles/m3). In fact, it
	  remains constant also, since we assume the ECF is too big to be
	  affected by anything.
	Vm[i] = current Vmem of cell #i in volts. It is a derived array -- we
	  recompute it frequently from cc_cells and z_array.

    ion_i is a dict that maps an ion name into its row index in the 2D
	arrays above; e.g., ion_i['K'] => 1.

What about gap junctions (GJs)?
    To get a network more complex than just isolated individual cells, we need
    GJs. These, too, are just stored in a big array:
	GJ_connects is a structured array [N_GJs]. Each element is one GJ.
          ['from']  and ['to'] specify the indices of its input and output cells
          (i.e., the column in the big arrays above).
    In addition to the network connectivity of the GJs, we must also set how
    quickly the various ions travel through them. This happens in three parts.
    - First GJ_connects['type'] tells you what type of GJ it is; different GJs
      may let different ions pass.
    - Next, GJ_diffusion[n_ions,type] gives the basic diffusion constant for
      each ion type through each GJ type in m^2/s.
    - Finally, as with ion channels, we must scale to account for the fraction
      of the cell membrane occupied by GJs. Specifically, GJ_connects['scale']
      lets you, for any individual GJ, scale all of the ions' diffusion
      constants identically. GJ_connects['scale'] is unitless and defaults to 1.

The gating system
    The gating system is how we implement gated ion channels, gated GJs,
    generation and decay. We can gate based on either ligand concentration or
    Vmem. In fact, for instructional purposes, we allow gatings that are
    completely "un-physical." E.g., the concentration of a ligand in one cell
    can affect an ion-channel diffusion constant in a completely different cell!

    The gating system is easily extensible. There are a few built-in gates
    (e.g., a ligand-gated Hill function and a proportional metabolite decay),
    but you can add as many new ones as you like (e.g., Hodgkin-Huxley gating
    for ion channels in neurons).

    At a top level, there are three lists: the ion-channel gates (reducing how
    fast ions flow through ICs), the GJ gates (ditto, for GJs), and the
    gen/decay gates (which create an ion's generation or decay, potentially
    depending on other inputs to, e.g., build a GRN).

    The next level is an individual gate object. Roughly it consists of:
    - the gating function. As noted, you can easily add new gating functions.
    - a parameter set. E.g., a Hill function might have Km, N and KVmax.

    Gating functions all deal with a 1D vector of inputs. Specifically, an IC or
    gen/decay function must deal with every cell at once; a GJ gating function
    must deal with every GJ at once. This vectoring is what makes the system run
    fast -- numpy vector operations are much faster than scalars. It is also
    biologically realistic -- e.g., a given type of ion channel is typically
    present to some degree in each cell. However, it does mean that if you want,
    e.g., to gate Na channels in only a few cells, then you must be clever;
    typically by picking parameter values for other cells that have the effect
    of creating a gate of unity value that does nothing.

    The parameter set for a gate is thus a 2D array with one row per parameter,
    and one column per cell (or GJ). Thus, while all the cells in a single
    gating share the same gating function, they may have different parameters
    per cell.

    A gating function produces a 1D vector of scale factors. For IC or gen/decay
    gating, the vector is [num_cells] and affects a single output ion across all
    cells. For GJ gating, the vector is [num_GJs] and affects GJ conductances,
    each of which affects all ions equally.

    The gating functions typically use either ligand concentrations or Vmem
    values as their inputs. Thus, sim_slopes() passes both cc_cells[] and Vm[]
    to the gating functions, which can then use either or both (or none, if they
    really want to). However, both cc_cells[] and Vm[] have width num_cells
    rather than num_GJs. This makes it hard for the gating function (which,
    again, is vectored) to do simple vector computations that combine an input
    vector[num_cells] with parameter vectors[num_GJs].

    The trick is that a gating function typically accesses (e.g.,) cc_cells with
    a line like "my_inputs = cc_cells[ligand_param]." If ligand_param is just a
    scalar, this grabs a vector[num_cells] of concentrations. But we can also
    use advanced indexing -- with, e.g., ligand_param=(2,[0,2,2,5]) -- to pull
    ion #2 and then pick/rearrange the values to create a vector[num_GJs]. In
    fact, we can even use this facility to do "non-physical" gating, where
    (e.g.,) an ion channel in one cell gates based on a ligand in a *different*
    cell.

    An end user typically builds a gating in one of two ways (e.g., for a Hill
    gating). You first call make_Hill_gates() to make an entire gate all at
    once. It takes a parameter for the gate destination (IC, GJ or gen/decay),
    and numerous other parameters. In most cases, this is the one-stop shopping
    function for making a gate. However, it makes an entire vector (i.e., the
    gating for every cell or GJ) all at once. You can give make_Hill_gates() a
    scalar for any given parameter (which assigns that value to every cell) or a
    vector (which can assign each cell its own value). However, the latter may
    not be convenient for the caller to set up.

    The other option is to (as always) first call make_Hill_gates(), typically
    giving it parameters that create unity gating (i.e., a do-nothing gate).
    Then call make_Hill_gate() as many times as needed, overriding the existing
    parameters for one cell with each call.
'''
import numpy as np
import scipy
import math	# To get pi.
import sim_toolbox as stb
import operator
import edebug as edb

#    import pdb; pdb.set_trace()

# Stuff all of the fundamental constants and commonly used parameters into a
# class. Any instance of this class will thus have all the constants.
class Params(object):
    def __init__(self):
        self.F = 96485  # Faraday constant [C/mol]
        self.R = 8.314  # Gas constant [J/K*mol]
        self.eo = 8.854e-12  # permittivity of free space [F/m]
        self.kb = 1.3806e-23  # Boltzmann constant [m2 kg/ s2 K1]
        self.q = 1.602e-19  # electron charge [C]
        self.tm = 7.5e-9  # thickness of cell membrane [nm]
        self.cm = 0.05  # patch capacitance of membrane [F/m2]
        self.T = 310  # temperature in Kelvin
        self.deltaGATP = -37000 # free energy released in ATP hydrolysis [J/mol]
        self.cATP = 1.5  # ATP concentration (mol/m3)
        self.cADP = 0.15  # ADP concentration (mol/m3)
        self.cPi = 0.15  # Pi concentration (mol/m3)
        self.alpha_NaK =1.0e-7 # max rate constant Na-K ATPase/unit surface area
        self.KmNK_Na = 12.0  # NaKATPase enzyme ext Na half-max sat value
        self.KmNK_K = 0.2  # NaKATPase enzyme ext K half-max sat value
        self.KmNK_ATP = 0.5  # NaKATPase enzyme ATP half-max sat value
        self.cell_r = 5.0e-6  # radius of single cell (m)
        self.GJ_len = 100e-9  # distance between two GJ connected cells (m)
        self.cell_sa = (4 * math.pi * self.cell_r ** 2)  # cell surface area
        self.cell_vol = ((4 / 3) * math.pi * self.cell_r ** 3)  # cell volume
        self.k26mV_inv = self.F / (self.R*self.T)

        # Simulation control. Sim_*_dump_interval is multiples of the .005 sec
        # timestep, and is only relevant for explicit integration.
        self.sim_dump_interval=10
        self.sim_long_dump_interval=100
        self.no_dumps = False

        # Numerical-integration parameters. These place a limit on how much
        # any cell's Vmem, or any ion concentration, can change in one timestep.
        self.sim_integ_max_delt_Vm = .0001	# Volts/step
        # .001 means that no [ion] can change by more than .1% in one timestep.
        self.sim_integ_max_delt_cc = .001
        self.adaptive_timestep = True	# So that the params above get used.
        self.use_implicit = False	# Use the implicit integrator
        self.max_step = 1e6		# big number
        self.rel_tol = 1e-3		# these are the default rel_tol and
        self.abs_tol = 1e-6		# abs_tol for solve_ivp()

def init_big_arrays (n_cells, n_GJs, p, extra_ions=[], n_GJ_types=1):
    global cc_cells, cc_env, Dm_array, z_array, ion_i, \
           GJ_diffusion, GJ_connects, GP, pump_scaling
    GP = p

    ### TEMPORARY CODE FOR PUMPS?
    pump_scaling = np.ones((n_cells))

    # ion properties (Name, base membrane diffusion [m2/s], valence
    #	initial concentration inside cell [mol/m3],
    #	fixed concentration outside cell [mol/m3], 
    # These are temporary structures. We use them to provide initial values for
    # the big arrays we are about to build, and to specify the order of which
    # row represents which ion in those arrays.
    Na={'Name':'Na', 'D_mem':1e-18, 'D_GJ':1e-18, 'z':1, 'c_in':10, 'c_out':145}
    K ={'Name':'K',  'D_mem':1e-18, 'D_GJ':1e-18, 'z':1, 'c_in':125,'c_out':5}
    Cl={'Name':'Cl', 'D_mem':1e-18, 'D_GJ':1e-18, 'z':-1,'c_in':55, 'c_out':140}
    P= {'Name':'P',  'D_mem':0,     'D_GJ':1e-18, 'z':-1,'c_in':80, 'c_out':10}

    # stack the above individual dictionaries into a list to make it easier to
    # process them in the loop below.
    ions_vect = [Na, K, Cl, P]

    # Any particular sim may want to declare extra ions.
    for ion in extra_ions:
        ions_vect.append ({'Name':ion, 'D_mem':0.0, 'D_GJ':1e-18,
                           'z':0, 'c_in':0,  'c_out':0})
    n_ions = len(ions_vect)

    cc_cells = np.empty((n_ions, n_cells))
    Dm_array = np.empty((n_ions, n_cells))
    z_array  = np.empty((n_ions))
    cc_env   = np.empty((n_ions))
    GJ_diffusion = np.empty((n_ions,n_GJ_types))

    ion_i = {}

    # Push the parameters of the above ions into the various arrays.
    for row, ion_obj in enumerate(ions_vect):
        cc_cells[row,:] = ion_obj['c_in']	# initial cell conc
        cc_env[row] = ion_obj['c_out']	# fixed environmental conc
        Dm_array[row] = ion_obj['D_mem']	# initial membrane diff coeff
        z_array[row] = ion_obj['z']		# fixed ion valence
        GJ_diffusion[row] = ion_obj['D_GJ']	# diffusion rate through GJs
        ion_i[ion_obj['Name']] = row		# map ion name -> its row

    # Create default arrays for GJs.
    GJ_connects=np.zeros((n_GJs), dtype=[('from','i4'),('to','i4'),
                                         ('scale','f4'),('type','i4')])

    # The global lists of gating objects.
    global IC_gates, GJ_gates, GD_gates
    IC_gates = []
    GJ_gates = []
    GD_gates = []

# The main "do-it" simulation function.
# Takes the current cc_cells[n_ions,n_cells], does all of the physics work, and
# returns an array of concentration slew rates [n_ions,n_cells]; i.e.,
# moles/m3 per second.
# Sim_slopes() is only called from sim.sim().
def sim_slopes (t, Cc_cells):
    global cc_env, Dm_array, z_array, ion_i, Vm, GJ_connects, GP, \
           cc_cells, pump_scaling
    cc_cells = Cc_cells
    Vm = compute_Vm (cc_cells)

    # 1. We will return slew_cc as (moles/m3) per sec. However, we first
    #    accumulate most of the fluxes as (moles/m2) per sec, and then multiply
    #    by the cell surface area later on.
    # 2. For (moles/m2)/s: m2 of what area? You might think that for, e.g., ion
    #    channels, it should be per m2 of ion-channel area -- but it's not.
    #    *All* fluxes are per m2 of cell-membrane area; that's what makes the
    #    system nice and consistent for not only ICs, but also GJs and pumps.
    #    Thus, the (e.g.,) diffusion rate through ion channels must be scaled
    #    down by the fraction of membrane area occupied by channels (and similar
    #    for ion pumps and GJs).
    slew_cc = np.zeros (cc_cells.shape) # Temporarily in (moles/m2) per sec.

    # Run the Na/K-ATPase ion pump in each cell.
    # Returns two 1D arrays[N_CELLS] of fluxes; units are (moles/m2)/s
    pump_Na,pump_K,_ = stb.pumpNaKATP(cc_cells[ion_i['Na']],cc_env[ion_i['Na']],
                                      cc_cells[ion_i['K']], cc_env[ion_i['K']],
                                      Vm, GP.T, GP, pump_scaling)

    # Kill the pumps on worm-interior cells (based on Dm=0 for all ions)
    keep_pumps = np.any (Dm_array>0, 0)	# array[n_cells]
    pump_Na *= keep_pumps
    pump_K  *= keep_pumps

    # Update the cell-interior [Na] and [K] after pumping (assume env is too big
    # to change its concentration).
    slew_cc[ion_i['Na']] = pump_Na
    slew_cc[ion_i['K']]  = pump_K

    f_GHK = stb.GHK (cc_env, cc_cells, Dm_array, z_array, Vm, GP)
    for g in IC_gates:
        f_GHK[g.out_ion] *= g.func (g, cc_cells, Vm, t)
    slew_cc += f_GHK	# again, still in (moles/m2) per sec

    # Gap-junction computations.
    deltaV_GJ = (Vm[GJ_connects['to']] - Vm[GJ_connects['from']]) # [n_GJs]

    # Get the gap-junction Norton-equivalent circuits for all ions at once.
    # Units of Ith are mol/(m2*s); units of Gth are mol/(m2*s) per Volt.
    (GJ_Ith, GJ_Gth) = GJ_norton(deltaV_GJ)	# Both are [n_ions,n_GJs].
    f_GJ = GJ_Ith + deltaV_GJ*GJ_Gth	# [n_ions,n_GJs]

    # g.func returns scale[n_GJs]; f_GJ is [n_ions,n_GJs]
    for g in GJ_gates:
        f_GJ *= g.func (g, cc_cells, Vm, t)

    # Update cells with GJ flux:
    # Note that the simple slew_cc[ion_index, GJ_connects['to']] += f_GJ
    # doesn't actually work in the case of two GJs driving the same 'to'
    # cell. Instead, we use np.add.at().
    for ion_index in range (cc_cells.shape[0]):	# for each ion
        np.add.at (slew_cc[ion_index,:], GJ_connects['from'], -f_GJ[ion_index])
        np.add.at (slew_cc[ion_index,:], GJ_connects['to'],    f_GJ[ion_index])

    # The current slew_cc units are moles/(m2*s), where the m2 is m2 of
    # cell-membrane area. To convert to moles/s entering the cell, we multiply
    # by the cell's surface area. Then, to convert to moles/m3 per s entering
    # the cell, we divide by the cell volume.
    slew_cc *= (GP.cell_sa / GP.cell_vol)

    # Next, do generation and decay. These are already natively in moles/(m3*s).
    for g in GD_gates:
        slew_cc[g.out_ion] += g.func (g, cc_cells, Vm, t)

    return (slew_cc)	# Moles/m3 per second.

# Given: per-cell, per-ion charges in moles/m3.
# First: sum them per-cell, scaled by valence to get "signed-moles/m3"
# Next: multiply by F to convert moles->coulombs. Multiply by cell volume/
# surface area to get coulombs/m2, and finally divide by Farads/m2.
# The final scaling factor is F * p.cell_vol / (p.cell_sa*p.cm),
# or about 3200 mV per (mol/m3)
def compute_Vm (Cc_cells):
    global cc_cells, GP
    cc_cells = Cc_cells

    # Calculate Vmem from scratch via the charge in the cells.
    rho_cells = (cc_cells * z_array[:,np.newaxis]).sum(axis=0) * GP.F
    return (rho_cells * GP.cell_vol / (GP.cell_sa*GP.cm))

# The main entry point into this file.
# Printout during simulation:
#	For explicit simulation, we print at intervals based on
#	sim_dump_interval and sim_long_dump_interval. For implicit simulation,
#	we don't print at all.
# Return values: (t_shots, cc_shots).
#	For both explicit and implicit simulation, we always take 100 snapshots
# during time [0,50] and then 200 shots over the entire rest of the simulation.
def sim (end_time):
    global cc_cells, Vm, GP
    if (GP.use_implicit):
        return (sim_implicit (end_time))

    # Save snapshots of core variables for plotting.
    t_shots=[]; cc_shots=[]; last_shot=-100;

    # run the simulation loop:
    i=0; t=0
    time_step = .005		# seconds
    while (t < end_time):
        slew_cc = sim_slopes(t,cc_cells)

        if (GP.adaptive_timestep):
            # Compute Vmem slew (in Volts/s). Essentially, it's just slew_Q/C.
            # Slew_cc is slew-flux in moles/m3 per second.
            # First, in each cell, sum all ions weighted by valence.
            # Then mpy by...
            #   ...(cell_vol/cell_sa) ->moles/(m2 of cell-memb cross-sec area)/s
            #   ...F -> Coul /(m2 of cell-membrane cross-sec area)/s
            #   ...1/C_mem -> Volts/s.
            mult = (GP.cell_vol / GP.cell_sa) * (GP.F/ GP.cm)
            slew_Vm = (slew_cc * z_array[:,np.newaxis]).sum(axis=0) * mult

            # Timestep control.
            # max_volts / (volts/sec) => max_time
            max_t_Vm = GP.sim_integ_max_delt_Vm / (np.absolute (slew_Vm).max())
            # (moles/m3*sec) / (moles/m3) => fractional_change / sec
            frac_cc = np.absolute(slew_cc)/(cc_cells+.00001)
            max_t_cc = GP.sim_integ_max_delt_cc / (frac_cc.max())
            n_steps = max (1, int (min (max_t_Vm, max_t_cc) / time_step))
            #print ('At t={}: max_t_Vm={}, max_t_cc={} => {} steps'.format(t, max_t_Vm, max_t_cc, n_steps))
            #print ('steps_Vm=', (.001/(time_step*np.absolute (slew_Vm))).astype(int))
        else:
            n_steps = 1

        # Dump out status occasionally during the simulation.
        # Note that this may be irregular; numerical integration could, e.g.,
        # repeatedly do i += 7; so if sim_dump_interval=10 we would rarely dump!
        if ((i % GP.sim_dump_interval == 0) and not GP.no_dumps):
            long = (i % GP.sim_long_dump_interval == 0)
            #edb.dump (t, cc_cells, edb.Units.mV_per_s, long)
            edb.dump (t, cc_cells, edb.Units.mol_per_m3s, long)
            #edb.analyze_equiv_network (cc_cells, GP)
            #edb.dump_gating ()

        if (t>9999999):		# A hook to stop & debug during a sim.
            edb.debug_print_GJ (GP, cc_cells, 1)
            import pdb; pdb.set_trace()
            print (sim_slopes (t, cc_cells))

        cc_cells +=  slew_cc * n_steps * time_step
        i += n_steps
        t = i*time_step

        # Save information for plotting at sample points. Early on (when things
        # are changing quickly) save lots of info. Afterwards, save seldom so
        # as to save memory (say 100 points before & 200 after)
        boundary=min (50,end_time);
        before=boundary/100; after=(end_time-boundary)/200
        interval = (before if t<boundary else after)
        if (t > last_shot+interval):
            t_shots.append(t)
            cc_shots.append(cc_cells.copy())
            last_shot = t

    return (t_shots, cc_shots)

# Replacement for sim(); it uses scipy.integrate.solve_ivp()
# Like sim(), it returns (t_shots, cc_shots).
def sim_implicit (end_time):
    import scipy.integrate
    global cc_cells, Vm, GP
    num_ions, num_cells = cc_cells.shape

    def wrap (t, y):
        global cc_cells
        #print ('----------------\nt={:.9g}'.format(t))
        #np.set_printoptions(formatter={'float':'{:6.2f}'.format},linewidth=120)
        #print ('y={}'.format(y))
        #print ("Vm = ", compute_Vm (y.reshape((num_ions,num_cells))))
        slew_cc = sim_slopes (t, y.reshape((num_ions,num_cells))) # moles/(m3*s)
        slew_cc = slew_cc.reshape (num_ions*num_cells)
        #np.set_printoptions(formatter={'float':'{:7.2g}'.format},linewidth=120)
        #print ('slews={}'.format(slew_cc))
        return (slew_cc)

    # Save information for plotting at sample points. Early on (when things
    # are changing quickly) save lots of info. Afterwards, save seldom so
    # as to save memory. So, 100 points in t=[0,50], then 200 in [50, end_time].
    boundary=min (50,end_time)
    t_eval = np.linspace (0,boundary,50,endpoint=False)
    if (end_time>50):
        t_eval = np.append (t_eval, np.linspace (boundary, end_time, 2000))

    # run the simulation loop
    y0 = cc_cells.reshape (num_ions*num_cells)
    bunch = scipy.integrate.solve_ivp (wrap, (0,end_time), y0, method='BDF',
                                       t_eval=t_eval, max_step=GP.max_step,
				       rtol=GP.rel_tol, atol=GP.abs_tol)
    print ('{} func evals, status={} ({}), success={}'.format \
             (bunch.nfev, bunch.status, bunch.message, bunch.success))
    if (not bunch.success):
        raise ValueError
    t_shots = t_eval.tolist()
    # bunch.y is [n_ions*n_cells, n_timepoints]
    cc_shots = [y.reshape((num_ions,num_cells)) for y in bunch.y.T]
    cc_cells = cc_shots[-1]
    return (t_shots, cc_shots)

# Builds and returns a Norton equivalent model for all GJs.
# Specifically, two arrays GJ_Ith and GJ_Gth of [n_ions,n_GJ].
# - GJ_Ith[i,g] is the diffusive flux of ion #i in the direction of
#   GJ[g].from->to, and has units (mol/m2*s). It ignores ion valence completely,
#   since it's just diffusion.
# - GJ_Gth*(Vto-Vfrom) is the drift flux of particles in the from->to direction;
#   GJ_Gth has units (mol/m2*s) per Volt. It is positive for negative ions and
#   vice versa.
# The caller uses them to compute flux (in moles/(m2*s), in the direction from
# GJ input to output, as flux = GJ_Ith + deltaV_GJ*GJ_Gth.
def GJ_norton (deltaV_GJ):
    global cc_cells, GP
    n_GJ = GJ_connects.size
    n_ions = cc_env.size

    # Compute ion drift and diffusion through GJs. Assume fixed GJ spacing
    # of GJ_len between connected cells.
    # First, compute d_conc/dx (assume constant conc in cells, and constant
    # gradients in the GJs).
    GJ_from = GJ_connects['from']	# Arrays of [n_GJ]
    GJ_to   = GJ_connects['to']
    D_scale = GJ_connects['scale']
    GJ_type = GJ_connects['type']

    deltaC_GJ=(cc_cells[:,GJ_to]-cc_cells[:,GJ_from])/GP.GJ_len #[n_ions,n_GJ]
    D = np.zeros ((n_ions, n_GJ))	# diffusion constants.
    conc = np.zeros ((n_ions, n_GJ))	# the concentration we use for drift

    for ion_index in range(n_ions):
        # Drift flux = velocity * conc. But a GJ connects two cells -- which
        # cell's concentration do we use? We originally used the average, but
        # that resulted in occasional negative concentrations. Now we use the
        # value of the cell that "sources" the ions (e.g., for a positive ion,
        # the cell with the more positive Vmem).
        # forw[n_GJ]: element #i is True if GJ[i] drifts forwards.
        forw = ((deltaV_GJ > 0) == (z_array[ion_index] < 0))
        # cell_idx[i] is the x such that GJ[i] uses cc[ion_index,x] for its conc
        cell_idx = np.where (forw, GJ_from, GJ_to)	# [n_GJ]
        # Now use advanced indexing to make conc[] for this ion, all GJs
        conc[ion_index] = cc_cells[ion_index,cell_idx]		# [n_GJ]

        # Pick the appropriate GJ diffusion constants for the GJ types.
        # Advanced indexing: GJ_diffusion is [n_ions,n_types] and GJ_type is
        # [n_GJs], so we get [n_GJs] * [n_GJs]
        D[ion_index] = GJ_diffusion[ion_index,GJ_type] * D_scale

    # GJ drift flux.
    u = D/(GP.kb*GP.T)	# drift mobility, by the Einstein relationship
    # [n_ions,n_GJ] * scalar * [n_ions] * [n_ions,n_GJ] = [n_ions,n_GJ]
    GJ_Gth = -conc * GP.q * z_array.reshape((n_ions,1)) * u / GP.GJ_len
    GJ_Ith = -D * deltaC_GJ
    return (GJ_Ith, GJ_Gth)

# Global arrays of Gate objects
IC_gates=[]
GJ_gates=[]
GD_gates=[]

# Constants to say which of the above global gates to put a new gate into.
GATE_IC=0; GATE_GJ=1; GATE_GD=2

# The runtime data structure for a single gate. Initialization parameters:
# - f: the gating function. All gates need a function.
# - dest: says whether this gate controls an ion channel, GJ or gen/decay.
#   It controls whether the gate gets appended to IC_gates,GJ_gates or GD_gates.
# - out_ion: if the gate affects an IC or gen/decay, then which ion it affects.
#   However, GJs gate all ions equally, and hence don't use this field (in which
#   case we don't set it, and nobody should look at it).
class Gate:
  def __init__(self, f, dest, out_ion):
    self.func = f
    if (dest != GATE_GJ):	# Used by the sim loop (see the comment
        self.out_ion = out_ion	# just above).
    self.width = (GJ_connects.size if (dest==GATE_GJ) else cc_cells.shape[1])
    gate_arrays = (IC_gates, GJ_gates, GD_gates)	# Stick this gate into
    gate_arrays[dest].append (self)			# the appropriate list.

######################################################
# Hill gating
######################################################
# This is the first function a user calls; it creates a new Hill gating for one
# output ion across all cells. It would most commonly be used for ion-channel
# gating or for generation/decay.
# You can supply kM, N and inv in many ways:
# - not at all, in which they default to unity gating; then fill the cells in
#   one by one later with make_Hill_gate(). Specifically, if you let 'inv'
#   default, then we get unity gates everywhere.
# - single numbers, which apply to all cells
# - a cell-by-cell array to gate each cell differently.
def make_Hill_gates (gtype,out_ion,in_ion,inv=None,kM=1,N=1,kVMax=1,offset=0.0):
    g = Gate (Hill_gating, gtype, out_ion)
    g.in_ion = in_ion	# Tell Hill_gating() which ligand controls the gate.
    if (inv==None):	# default everything to unity gating
        inv=True; kM=1e30	# i.e., kVMax * (1 / (1 - 0**N)) with kVMax=1.
    g.params = np.zeros ((5,g.width))	# (5 params) x (# cells or GJs)
    g.params[0],g.params[1],g.params[2],g.params[3],g.params[4]=inv,kM,N,kVMax,offset
    return (g)

# Changes the parameter of an existing gate 'g' in just one 'cell'.
def make_Hill_gate (g, cell, inv, kM=1, kVMax=1, N=1, offset=0.0):
    g.params[0:5,cell] = [inv, kM, N, kVMax, offset]

# If a ligand in one cell is to affect a gate in a *different* cell, then
# Hill_gate_unphysical_inputs() uses advanced indexing to let us do so.
# The trick: normally in_ion is simply the number of which ion gates us, and
# Hill_gating() gets its vector of inputs with cc_cells[in_ion]. So if in_ion=1,
# then cc_cells[1] is [K] for all cells.
# However, Hill_gate_unphysical_inputs() makes in_ion into a tuple; e.g.,
# (1, [2,3,3,0,0]) for a 5-cell system. Then cc_cells[(1, [2,3,3,0,0])] gives us
# cc_cells[1][2,3,3,0,0], which first gets cc_cells[1] and then uses the trick
# of indexing a vector with another vector to give us a *reordering* of
# cc_cells[1]: in this case, it is [1,2], [1,3], [1,3], [1,0], [1,0].
# Unphysical gating is *needed* anytime we want to gate GJs rather than ICs
# (see the comment at the top of the file for more details).
def Hill_gate_unphysical_inputs (g, inputs):
    g.in_ion=tuple([g.in_ion,inputs])

# The function that actually computes the gating vector at runtime.
def Hill_gating (g, cc, Vm_ignore, t_ignore):
    # g.params[] is [5, # cells or GJs]. Assign each of its rows (i.e., a vector
    # of one parameter across all cells) to one variable.
    inv,kM,N,kVMax,offset = g.params # Each var is a vector (one val per cell).
    conc = cc[g.in_ion]	# vector [N_CELLS]
    out = np.array (1 / (1 + ((conc/kM)**N)))
    out = 1 / (1 + ((conc/kM)**N))
    out = kVMax*((inv*out + (1-inv)*(1-out)) + offset)	# any better way to do this?
    return (out)

######################################################
# Vmem gating
######################################################
# See all of the comments above for Hill-function gating.

# Force to 1 with kM=big and kVMax=1
def make_Vmem_gates (gtype, out_ion, kM=100, N=1, kVMax=1):
    g = Gate (Vmem_gating, gtype, out_ion)
    g.in_cells = []
    g.params = np.zeros ((3,g.width))
    g.params[0], g.params[1], g.params[2] = kM, N, kVMax
    return (g)

def make_Vmem_gate (g, cell, kM, N=1, kVMax=1):
    g.params[0:3,cell] = kM, N, kVMax

def Vmem_gate_unphysical_inputs (g, inputs):
    g.in_cells = inputs

def Vmem_gating (g, cc_ignore, Vm, t_ignore):
    kM,N,kVMax = g.params	# assign each row to one variable
    V = Vm[g.in_cells]
    out = kVMax / (1 + np.exp (N * (V-kM)))
    return (out)

######################################################
# Const-gen/decay gating
######################################################
# See all of the comments above for Hill-function gating.
# Gen is a simple constant generation rate (moles/m^3 per s); decay is a simple
# constant decay rate in 1/s.
# If you want Hill-function generation, then this gate isn't enough. You could
# use a Hill-function gate with gtype=GATE_GD; then you could instantiate a
# GD_const_gating with gen=0 to get the decay.

def make_GD_const_gates (gtype, out_ion, gen=0, decay=0):
    assert (gtype==GATE_GD)	# There's no other sensible choice.
    g = Gate (GD_const_gating, gtype, out_ion=out_ion)
    g.in_ion=out_ion
    g.params = np.zeros ((2,g.width))
    g.params[0], g.params[1] = gen, decay
    return (g)

def make_GD_const_gate (g, cell, gen=0, decay=0):
    g.params[0:2, cell] = [gen, decay]

def GD_const_gating (g, cc, Vm_ignore, t_ignore):
    gen_rate,decay_rate = g.params	# Each is vector[N_CELLS]
    out = gen_rate - cc[g.in_ion]*decay_rate
    return (out)