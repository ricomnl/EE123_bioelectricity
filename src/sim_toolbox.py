#!/usr/bin/env python3
# Copyright 2014-2018 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.


import numpy as np
import numpy.ma as ma
from scipy import interpolate as interp
from scipy.ndimage.filters import gaussian_filter


# Toolbox of functions used in the Simulator class to calculate key bioelectric properties.

# This is the new/improved version of GHK(). The only difference is that it
# works on entire 2D arrays at once, rather than one ion at a time.
def GHK (cExt,cIn,D,zc,Vmem,p):
    """
    Goldman-Hodges-Katz between two connected volumes.
    Return the flux moving *into* the cell

    This function simplifies to regular diffusion if Vba == 0.0.

    This function takes numpy matrix values as input.

    This is the Goldman Flux/Current Equation (not to be confused with the
    Goldman Equation). Note: the Nernst-Planck equation has been trialed in
    place of this, and it does not reproduce proper reversal potentials.

    Parameters
    ----------
    cExt[n_ions]	concentration in the ECF [moles/m3]
    cIn[n_ions,n_cells]	concentration in the ICF [moles/m3]
    D[n_ions,n_cells]	Diffusion constant of the ion  [m2/s]
    zc[n_ions]		valence of the ion
    Vmem[n_cells]	voltage difference vIn - vOut
    p			an instance of the Parameters class

    Returns
    --------
    flux[n_ions,n_cells] Chemical flux magnitude flowing *into* the cell [mol/s]

    """

    # GHK becomes 0/0 when zVmem=0. Avoid this by trying to never encounter
    # Vmem=0 or z=0.
    FLOAT_NONCE = 1.0e-25
    Vmem += FLOAT_NONCE			# [c]
    zc += FLOAT_NONCE			# [i]

    # [i,1]*[c,1]=[i,c]
    i = zc.size; c=Vmem.size
    ZVrel = np.reshape(zc,(i,1))*np.reshape(Vmem*p.k26mV_inv,(1,c))

    exp_ZVrel = np.exp(-ZVrel)		# exp (-Z*Vrel)
    deno = -np.expm1(-ZVrel)		# 1 - exp(-Z*Vrel), the GHK denominator
    # tm = thickness of the cell membrane [meters]
    P = D / p.tm 			# permeability of a channel = D/L; [i,c]

    # -P*Z*Vrel * (cIn - cExt*exp(-Z*Vrel)) / (1 - exp(-Z*Vrel))
    cExt_exp = (cExt.T * exp_ZVrel.T).T
    flux = -P*ZVrel*((cIn - cExt_exp)/deno)

    return flux

# This is the old (and unused) version of GHK. It takes 1D inputs.
def GHK_old(cExt,cIn,D,zc,Vmem,p):
    """
    Goldman-Hodges-Katz between two connected volumes.
    Return the flux moving *into* the cell

    This function defaults to regular diffusion if Vba == 0.0.

    This function takes numpy matrix values as input. All inputs must be
    matrices of the same shape.

    This is the Goldman Flux/Current Equation (not to be confused with the
    Goldman Equation). Note: the Nernst-Planck equation has been trialed in
    place of this, and it does not reproduce proper reversal potentials.

    Parameters
    ----------
    cExt	concentration in the ECF [moles/m3]
    cIn		concentration in the ICF [moles/m3]
    D		Diffusion constant of the ion  [m2/s]
    zc		valence of the ion
    Vmem	voltage difference vIn - vOut
    p           an instance of the Parameters class

    Returns
    --------
    flux        Chemical flux magnitude flowing *into* the cell [mol/s]

    """

    # GHK becomes 0/0 when Vmem=0. Avoid this by trying to never encounter
    # Vmem=0. I'm not sure why Alexis added FLOAT_NONCE to zc also.
    FLOAT_NONCE = 1.0e-25
    Vmem += FLOAT_NONCE
    zc += FLOAT_NONCE

    ZVrel = zc*Vmem* p.k26mV_inv	# Z * (Vmem/26mV), or Z*Vrel
    exp_ZVrel = np.exp(-ZVrel)		# exp (-Z*Vrel)
    deno = -np.expm1(-ZVrel)		# 1 - exp(-Z*Vrel), the GHK denominator
    # tm= thickness of the cell membrane [m]
    P = D / p.tm			# permeability of a channel = D/L

    # -P*Z*Vrel * (cIn - cExt*exp(-Z*Vrel)) / (1 - exp(-Z*Vrel))
    flux = -P*ZVrel*((cIn -cExt*exp_ZVrel)/deno)

    return flux

def pumpNaKATP(cNai,cNao,cKi,cKo,Vm,T,p,block, met = None):

    """
    Parameters
    ----------
    cNai            Concentration of Na+ inside the cell
    cNao            Concentration of Na+ outside the cell
    cKi             Concentration of K+ inside the cell
    cKo             Concentration of K+ outside the cell
    Vm              Voltage across cell membrane [V]
    p               An instance of Parameters object

    met             A "metabolism" vector containing concentrations of ATP, ADP and Pi


    Returns
    -------
    f_Na            Na+ flux (into cell +)
    f_K             K+ flux (into cell +)
    """

    deltaGATP_o = p.deltaGATP  # standard free energy of ATP hydrolysis reaction in J/(mol K)

    cATP = p.cATP
    cADP = p.cADP
    cPi  = p.cPi

    # calculate the reaction coefficient Q:
    Qnumo = (cADP*1e-3)*(cPi*1e-3)*((cNao*1e-3)**3)*((cKi*1e-3)**2)
    Qdenomo = (cATP*1e-3)*((cNai*1e-3)**3)*((cKo*1e-3)** 2)

    # ensure no chance of dividing by zero:
    inds_Z = (Qdenomo == 0.0).nonzero()
    Qdenomo[inds_Z] = 1.0e-15

    Q = Qnumo / Qdenomo


    # calculate the equilibrium constant for the pump reaction:
    Keq = np.exp(-(deltaGATP_o / (p.R * T) - ((p.F * Vm) / (p.R * T))))

    # calculate the enzyme coefficient:
    numo_E = ((cNai/p.KmNK_Na)**3) * ((cKo/p.KmNK_K)**2) * (cATP/p.KmNK_ATP)
    denomo_E = (1 + (cNai/p.KmNK_Na)**3)*(1+(cKo/p.KmNK_K)**2)*(1+(cATP/p.KmNK_ATP))

    fwd_co = numo_E/denomo_E

    f_Na = -3*block*p.alpha_NaK*fwd_co*(1 - (Q/Keq))  # flux as [mol/m2s]   scaled to concentrations Na in and K out

    f_K = -(2/3)*f_Na          # flux as [mol/m2s]

    return f_Na, f_K, -f_Na  # FIXME get rid of this return of extra -f_Na!!

def pumpCaATP(cCai,cCao,Vm,T,p, block, met = None):

    """
    Parameters
    ----------
    cCai            Concentration of Ca2+ inside the cell
    cCao            Concentration of Ca2+ outside the cell
    voli            Volume of the cell [m3]
    volo            Volume outside the cell [m3]
    Vm              Voltage across cell membrane [V]
    p               An instance of Parameters object


    Returns
    -------
    cCai2           Updated Ca2+ inside cell
    cCao2           Updated Ca2+ outside cell
    f_Ca            Ca2+ flux (into cell +)
    """


    deltaGATP_o = p.deltaGATP

    no_negs(cCai)
    no_negs(cCao)

    cATP = p.cATP
    cADP = p.cADP
    cPi  = p.cPi
    #

    # calculate the reaction coefficient Q:
    Qnumo = cADP * cPi * cCao
    Qdenomo = cATP * cCai

    # ensure no chance of dividing by zero:
    inds_Z = (Qdenomo == 0.0).nonzero()
    Qdenomo[inds_Z] = 1.0e-16

    Q = Qnumo / Qdenomo

    # calculate the equilibrium constant for the pump reaction:
    Keq = np.exp(-(deltaGATP_o / (p.R * T) - 2*((p.F * Vm) / (p.R * T))))

    # calculate the enzyme coefficient for forward reaction:
    numo_E = (cCai/p.KmCa_Ca) * (cATP/p.KmCa_ATP)
    denomo_E = (1 + (cCai/p.KmCa_Ca)) * (1+ (cATP/p.KmCa_ATP))

    frwd = numo_E/denomo_E

    # calculate the enzyme coefficient for backward reaction:
    numo_Eb = (cCao/p.KmCa_Ca)
    denomo_Eb = (1 + (cCao/p.KmCa_Ca))

    bkwrd = numo_Eb/denomo_Eb

    f_Ca = -p.alpha_Ca*frwd*(1 - (Q/Keq))  # flux as [mol/m2s]

    return f_Ca

def pumpCaER(cCai,cCao,Vm,T,p):
    """
    Pumps calcium out of the cell and into the endoplasmic reticulum.
    Vm is the voltage across the endoplasmic reticulum membrane.

    """

    deltaGATP_o = p.deltaGATP

    cATP = p.cATP
    cADP = p.cADP
    cPi = p.cPi

    # calculate the reaction coefficient Q:
    Qnumo = cADP * cPi * cCai
    Qdenomo = cATP * cCao

    # ensure no chance of dividing by zero:
    inds_Z = (Qdenomo == 0.0).nonzero()
    Qdenomo[inds_Z] = 1.0e-16

    Q = Qnumo / Qdenomo

    # calculate the equilibrium constant for the pump reaction:
    Keq = np.exp(-deltaGATP_o / (p.R * T) - 2 * ((p.F * Vm) / (p.R * T)))


    # calculate the enzyme coefficient for forward reaction:
    numo_E = (cCao / p.KmCa_Ca) * (cATP / p.KmCa_ATP)
    denomo_E = (1 + (cCao / p.KmCa_Ca)) * (1 + (cATP / p.KmCa_ATP))

    frwd = numo_E / denomo_E

    # calculate the enzyme coefficient for backward reaction:
    numo_Eb = (cCai / p.KmCa_Ca)
    denomo_Eb = (1 + (cCai / p.KmCa_Ca))

    bkwrd = numo_Eb / denomo_Eb

    f_Ca = p.serca_max * frwd * (1 - (Q / Keq))  # flux as [mol/m2s]

    return f_Ca

def check_v(vm):
    """
    Does a quick check on Vmem values
    and displays error warning or exception if the value
    indicates the simulation is unstable.

    """


    isnans = np.isnan(vm)

    if isnans.any():  # if there's anything in the isubzeros matrix...
        raise BetseSimInstabilityException(
            "Your simulation has become unstable. Please try a smaller time step,"
            "reduce gap junction radius, and/or reduce pump rate coefficients.")

def nernst_planck_flux(c, gcx, gcy, gvx, gvy,ux,uy,D,z,T,p):
    """
     Calculate the flux component of the Nernst-Planck equation

     Parameters
     ------------

    c:     concentration
    gcx:   concentration gradient, x component
    gcy:   concentration gradient, y component
    gvx:   voltage gradient, x component
    gvy:   voltage gradient, y component
    ux:    fluid velocity, x component
    uy:    fluid velocity, y component
    D:     diffusion constant, D
    z:     ion charge
    T:     temperature
    p:     parameters object

    Returns
    --------
    fx, fx        mass flux in x and y directions
    """

    alpha = (D*z*p.q)/(p.kb*T)
    fx =  -D*gcx - alpha*gvx*c + ux*c
    fy =  -D*gcy - alpha*gvy*c + uy*c

    return fx, fy

def nernst_planck_vector(c, gc, gv,u,D,z,T,p):
    """
     Calculate the flux component of the Nernst-Planck equation
     along a directional gradient (e.g. gap junction)

     Parameters
     ------------

    c:     concentration
    gc:   concentration gradient
    gv:   voltage gradient
    u:    fluid velocity
    D:     diffusion constant, D
    z:     ion charge
    T:     temperature
    p:     parameters object

    Returns
    --------
    fx, fx        mass flux in x and y directions
    """

    alpha = (D*z*p.q)/(p.kb*T)
    f =  -D*gc - alpha*gv*c + u*c

    return f

def no_negs(data):
    """
    This function screens an (concentration) array to
    ensure there are no NaNs and no negative values,
    crashing with an instability message if it finds any.

    """

    # ensure no NaNs:
    inds_nan = (np.isnan(data)).nonzero()

    # ensure that data has no less than zero values:
    inds_neg = (data < 0.0).nonzero()

    if len(inds_neg[0]) > 0:

        data[inds_neg] = 0.0 # add in a small bit to protect from crashing

    if len(inds_nan[0]) > 0:

        raise BetseSimInstabilityException(
            "Your simulation has become unstable. Please try a smaller time step,"
            "reduce gap junction radius, and/or reduce rate coefficients.")

    return data

def bicarbonate_buffer(cCO2, cHCO3):
    """
    This most amazing buffer handles influx of H+,
    HCO3-, H2CO3 (from dissolved carbon dioxide) to
    handle pH in real time.

    Uses the bicarbonate dissacociation reaction:

    H2CO3 ----> HCO3 + H

    Where all dissolved carbon dioxide is assumed
    converted to carbonic acid via carbonic anhydrase enzyme.

    """

    pH = 6.1 + np.log10(cHCO3/cCO2)

    cH = 10**(-pH)*1e3

    return cH, pH

def molecule_pump(sim, cX_cell_o, cX_env_o, cells, p, Df=1e-9, z=0, pump_into_cell =False, alpha_max=1.0e-8, Km_X=1.0,
                 Km_ATP=1.0, met = None, n=1, ignoreECM = True, rho = 1.0):


    """
    Defines a generic active transport pump that can be used to move
    a general molecule (such as serotonin or glutamate)
    into or out of the cell by active transport.

    Works on the basic premise of enzymatic pumps defined elsewhere:

    pump_out is True:

    cX_cell + cATP  -------> cX_env + cADP + cPi

    pump_out is False:

    cX_env + cATP  <------- cX_cell + cADP + cPi

    Parameters
    -------------
    cX_cell_o           Concentration of X in the cell         [mol/m3]
    cX_env_o            Concentration of X in the environment  [mol/m3]
    cells               Instance of cells
    p                   Instance of parameters
    z                   Charge of X
    pump_out            Is pumping out of cell (pump_out = True) or into cell (pump_out = False)?
    alpha_max           Maximum rate constant of pump reaction [mol/s]
    Km_X                Michaelis-Mentin 1/2 saturation value for X [mol/m3]
    Km_ATP              Michaelis-Mentin 1/2 saturation value for ATP [mol/m3]

    Returns
    ------------
    cX_cell_1     Updated concentration of X in cells
    cX_env_1      Updated concentration of X in environment
    f_X           Flux of X (into the cell +)

    """

    deltaGATP_o = p.deltaGATP  # standard free energy of ATP hydrolysis reaction in J/(mol K)

    if met is None:

        # if metabolism vector not supplied, use singular defaults for concentrations
        cATP = p.cATP
        cADP = p.cADP
        cPi  = p.cPi

    else:

        cATP = met['cATP']  # concentration of ATP in mmol/L
        cADP = met['cADP']  # concentration of ADP in mmol/L
        cPi = met['cPi']  # concentration of Pi in mmol/L

    if p.is_ecm is True:

        cX_env = cX_env_o[cells.map_mem2ecm]

        cX_cell = cX_cell_o[cells.mem_to_cells]

    else:
        cX_env = cX_env_o[:]
        cX_cell = cX_cell_o[cells.mem_to_cells]

    if pump_into_cell is False:

        # active pumping of molecule from cell and into environment:
        # calculate the reaction coefficient Q:
        Qnumo = cADP * cPi * (cX_env**n)
        Qdenomo = cATP * (cX_cell**n)

        # ensure no chance of dividing by zero:
        inds_Z = (Qdenomo == 0.0).nonzero()
        Qdenomo[inds_Z] = 1.0e-10

        Q = Qnumo / Qdenomo

        # calculate the equilibrium constant for the pump reaction:
        Keq = np.exp(-deltaGATP_o / (p.R * sim.T) + ((n*z * p.F * sim.vm) / (p.R * sim.T)))

        # calculate the reaction rate coefficient
        alpha = alpha_max * (1 - (Q / Keq))

        # calculate the enzyme coefficient:
        numo_E = ((cX_cell / Km_X)**n) * (cATP / Km_ATP)
        denomo_E = (1 + (cX_cell / Km_X)**n) * (1 + (cATP / Km_ATP))

        f_X = -rho*alpha * (numo_E / denomo_E)  # flux as [mol/m2s]   scaled to concentrations Na in and K out

    else:

        # active pumping of molecule from environment and into cell:
        # calculate the reaction coefficient Q:
        Qnumo = cADP * cPi * cX_cell
        Qdenomo = cATP * cX_env

        # ensure no chance of dividing by zero:
        inds_Z = (Qdenomo == 0.0).nonzero()
        Qdenomo[inds_Z] = 1.0e-10

        Q = Qnumo / Qdenomo

        # calculate the equilibrium constant for the pump reaction:
        Keq = np.exp(-deltaGATP_o / (p.R * sim.T) - ((z * p.F * sim.vm) / (p.R * sim.T)))

        # calculate the reaction rate coefficient
        alpha = alpha_max * (1 - (Q / Keq))

        # calculate the enzyme coefficient:
        numo_E = (cX_env / Km_X) * (cATP / Km_ATP)
        denomo_E = (1 + (cX_env / Km_X)) * (1 + (cATP / Km_ATP))

        f_X = rho* alpha * (numo_E / denomo_E)  # flux as [mol/m2s]   scaled to concentrations Na in and K out

    if p.cluster_open is False:
        f_X[cells.bflags_mems] = 0

    cmems = cX_cell_o[cells.mem_to_cells]

    # update cell and environmental concentrations
    cX_cell_1, _, cX_env_1 = update_Co(sim, cX_cell_o, cmems, cX_env_o, f_X, cells, p, ignoreECM = ignoreECM)


    if p.is_ecm is False:
        cX_env_1_temp = cX_env_1.mean()
        cX_env_1[:] = cX_env_1_temp

    return cX_cell_1, cX_env_1, f_X

def molecule_transporter(sim, cX_cell_o, cX_env_o, cells, p, Df=1e-9, z=0, pump_into_cell=False, alpha_max=1.0e-8,
        Km_X=1.0, Keq=1.0, n = 1.0, ignoreECM = True, rho = 1.0):


    """
    Defines a generic facillitated transporter that can be used to move
    a general molecule (such as glucose).

    ATP is not used for the transporter

    Works on the basic premise of enzymatic pumps defined elsewhere:

    pump_out is True:

    cX_cell  -------> cX_env

    pump_out is False:

    cX_env   <------- cX_cell

    Parameters
    -------------
    cX_cell_o           Concentration of X in the cell         [mol/m3]
    cX_env_o            Concentration of X in the environment  [mol/m3]
    cells               Instance of cells
    p                   Instance of parameters
    z                   Charge of X
    pump_out            Is pumping out of cell (pump_out = True) or into cell (pump_out = False)?
    alpha_max           Maximum rate constant of pump reaction [mol/s]
    Km_X                Michaelis-Mentin 1/2 saturation value for X [mol/m3]

    Returns
    ------------
    cX_cell_1     Updated concentration of X in cells
    cX_env_1      Updated concentration of X in environment
    f_X           Flux of X (into the cell +)

    """

    if p.is_ecm is True:

        cX_env = cX_env_o[cells.map_mem2ecm]

        cX_cell = cX_cell_o[cells.mem_to_cells]

    else:
        cX_env = cX_env_o
        cX_cell = cX_cell_o[cells.mem_to_cells]

    if pump_into_cell is False:

        # active pumping of molecule from cell and into environment:
        # calculate the reaction coefficient Q:
        Qnumo = cX_env
        Qdenomo = cX_cell

        # ensure no chance of dividing by zero:
        inds_Z = (Qdenomo == 0.0).nonzero()
        Qdenomo[inds_Z] = 1.0e-15

        Q = Qnumo / Qdenomo

        # modify equilibrium constant by membrane voltage if ion is charged:
        Keq = Keq*np.exp((z * p.F * sim.vm) / (p.R * sim.T))

        # calculate the reaction rate coefficient
        alpha = alpha_max * (1 - (Q / Keq))

        # calculate the enzyme coefficient:
        numo_E = (cX_cell / Km_X)
        denomo_E = (1 + (cX_cell / Km_X))

        f_X = -rho*alpha * (numo_E / denomo_E)  # flux as [mol/m2s]   scaled to concentrations Na in and K out


    else:

        # active pumping of molecule from environment and into cell:
        # calculate the reaction coefficient Q:
        Qnumo = cX_cell
        Qdenomo = cX_env

        # ensure no chance of dividing by zero:
        inds_Z = (Qdenomo == 0.0).nonzero()
        Qdenomo[inds_Z] = 1.0e-15

        Q = Qnumo / Qdenomo

        # modify equilibrium constant by membrane voltage if ion is charged:
        Keq = Keq*np.exp(-(z * p.F * sim.vm) / (p.R * sim.T))

        # calculate the reaction rate coefficient
        alpha = alpha_max * (1 - (Q / Keq))

        # calculate the enzyme coefficient:
        numo_E = (cX_env / Km_X)
        denomo_E = (1 + (cX_env / Km_X))

        f_X = rho*alpha * (numo_E / denomo_E)  # flux as [mol/m2s]   scaled to concentrations Na in and K out

    if p.cluster_open is False:
        f_X[cells.bflags_mems] = 0

    cmems = cX_cell_o[cells.mem_to_cells]

    # update cell and environmental concentrations
    cX_cell_1, _, cX_env_1 = update_Co(sim, cX_cell_o, cmems, cX_env_o, f_X, cells, p, ignoreECM= ignoreECM)

    # next electrodiffuse concentrations around the cell interior:
    # cX_cell_1 = update_intra(sim, cells, cX_cell_1, Df, z, p)

    # ensure that there are no negative values
    # cX_cell_1 = no_negs(cX_cell_1)
    # cX_env_1 = no_negs(cX_env_1)


    if p.is_ecm is False:
        cX_env_1_temp = cX_env_1.mean()
        cX_env_1[:] = cX_env_1_temp

    return cX_cell_1, cX_env_1, f_X

def molecule_mover(sim, cX_env_o, cX_cells, cells, p, z=0, Dm=1.0e-18, Do=1.0e-9, Dgj=1.0e-12, Ftj = 1.0, c_bound=0.0,
                   ignoreECM = True, smoothECM = False, ignoreTJ = False, ignoreGJ = False, rho = 1, cmems = None,
                   time_dilation_factor = 1.0, update_intra = False, name = "Unknown"):

    """
    Transports a generic molecule across the membrane,
    through gap junctions, and if p.is_ecm is true,
    through extracellular spaces and the environment.

    Parameters
    -----------
    cX_cell_o           Concentration of molecule in the cytosol [mol/m3]
    cX_env_o            Concentration of molecule in the environment [mol/m3]
    cells               Instance of Cells
    p                   Instance of Parameters
    z                   Charge state of molecule
    Dm                  Membrane diffusion constant [m2/s]
    Do                  Free diffusion constant [m2/s]
    Ftj                 Factor influencing relative diffusion of substance across tight junction barrier
    c_bound             Concentration of molecule at global bounds (required for is_ecm True only)

    Returns
    -----------
    cX_cell_1         Updated concentration of molecule in the cell
    cX_env_1          Updated concentration of molecule in the environment

    """


    if p.is_ecm is True:

        cX_env = cX_env_o[cells.map_mem2ecm]

    else:
        cX_env = cX_env_o[:]

    if cmems is None:

        cX_mems = cX_cells[cells.mem_to_cells]

    elif len(cmems) == sim.mdl:
        cX_mems = cmems

    Dm_vect = np.ones(len(cX_mems))*Dm

    # Transmembrane: electrodiffuse molecule X between cell and extracellular space------------------------------

    if Dm != 0.0:  # if there is some finite membrane diffusivity, exchange between cells and env space:

        IdM = np.ones(sim.mdl)

        f_X_ED = electroflux(cX_env, cX_mems, Dm_vect, p.tm*IdM, z*IdM, sim.vm, sim.T, p, rho = rho)

        if p.cluster_open is False:
            f_X_ED[cells.bflags_mems] = 0

    else:

        f_X_ED = np.zeros(sim.mdl)

    # ------------------------------------------------------------
    if ignoreGJ is False:

        fgj_X = electroflux(cX_mems[cells.mem_i],
                       cX_mems[cells.nn_i],
                       Dgj*sim.gj_block*sim.gjopen,
                       cells.gj_len*np.ones(sim.mdl),
                       z*np.ones(sim.mdl),
                       sim.vgj,
                       p.T,
                       p,
                       rho=1
                       )


        # enforce zero flux at outer boundary:
        fgj_X[cells.bflags_mems] = 0.0

        # divergence calculation for individual cells (finite volume expression)
        delta_cco = np.dot(cells.M_sum_mems, -fgj_X*cells.mem_sa) / cells.cell_vol


        # Calculate the final concentration change (the acceleration effectively speeds up time):

        if update_intra is False: # do the GJ transfer assuming instant mixing in the cell:
            cX_cells = cX_cells + p.dt*delta_cco*time_dilation_factor
            cX_mems = cX_cells[cells.mem_to_cells]

        else: # only do the GJ transfer to the membrane domains:
            cX_mems = cX_mems - fgj_X*p.dt*(cells.mem_sa/cells.mem_vol)*time_dilation_factor


    else:
        fgj_X = np.zeros(sim.mdl)

    # update concentrations due to electrodiffusion, updated at the end to do as many updates in one piece as possible:
    cX_cells, cX_mems, cX_env_o = update_Co(sim, cX_cells, cX_mems, cX_env_o, f_X_ED, cells, p, ignoreECM=ignoreECM)


    #------------------------------------------------------------------------------------------------------------

    # Transport through environment, if p.is_ecm is True-----------------------------------------------------

    if p.is_ecm is True:

        env_check = len((cX_env_o != 0.0).nonzero()[0])

        if  env_check != 0.0 or c_bound > 1.0e-15:

            cenv = cX_env_o
            cenv = cenv.reshape(cells.X.shape)

            cenv[:, 0] = c_bound
            cenv[:, -1] = c_bound
            cenv[0, :] = c_bound
            cenv[-1, :] = c_bound

            denv_multiplier = np.ones(len(cells.xypts))

            if ignoreTJ is False:
                # if tight junction barrier applies, create a mask that defines relative strength of barrier:
                denv_multiplier = denv_multiplier.reshape(cells.X.shape)*sim.D_env_weight

                # at the cluster boundary, further modify the env diffusion map by a relative TJ diffusion factor:
                denv_multiplier.ravel()[sim.TJ_targets] = denv_multiplier.ravel()[sim.TJ_targets]*Ftj

            else:
                denv_multiplier = denv_multiplier.reshape(cells.X.shape)

            gcx, gcy = fd.gradient(cenv, cells.delta)

            if p.fluid_flow is True:

                ux = sim.u_env_x.reshape(cells.X.shape)
                uy = sim.u_env_y.reshape(cells.X.shape)


            else:

                ux = 0.0
                uy = 0.0

            fx, fy = nernst_planck_flux(cenv, gcx, gcy, -sim.E_env_x, -sim.E_env_y, ux, uy,
                                            denv_multiplier*Do, z, sim.T, p)

            div_fa = fd.divergence(-fx, -fy, cells.delta, cells.delta)

            fenvx = fx
            fenvy = fy

            cenv = cenv + div_fa * p.dt*time_dilation_factor


            # if p.sharpness < 1.0:
            #
            #     cenv = fd.integrator(cenv, sharp = p.sharpness)

            cX_env_o = cenv.ravel()

        else:

            fenvx = np.zeros(sim.edl)
            fenvy = np.zeros(sim.edl)

    else:
        cX_env_temp = cX_env_o.mean()
        cX_env_o = np.zeros(sim.mdl)
        cX_env_o[:] = cX_env_temp
        fenvx = 0
        fenvy = 0

    # check for sub-zero concentrations:
    indsZm = (cX_mems < 0.0).nonzero()[0]

    if len(indsZm) > 0:
        raise BetseSimInstabilityException(
            "Network concentration of " + name + " on membrane below zero! Your simulation has"
                                                   " become unstable.")
    indsZc = (cX_cells < 0.0).nonzero()[0]

    if len(indsZc) > 0:
        raise BetseSimInstabilityException(
            "Network concentration of " + name + " in cells below zero! Your simulation has"
                                                   " become unstable.")

    indsZe = (cX_env_o < 0.0).nonzero()[0]

    if len(indsZe) > 0:
        print(name, c_bound)
        raise BetseSimInstabilityException(
            "Network concentration of " + name + " in environment below zero! Your simulation has"
                                                   " become unstable.")

    # variable summing all sub-zero array checks
    # lencheck = len(indsZm) + len(indsZc) + len(indsZe)

    # if lencheck > 0:
    #     raise BetseSimInstabilityException(
    #         "Network concentration of " + name + " below zero! Your simulation has"
    #                                                " become unstable.")


    return cX_env_o, cX_cells, cX_mems, f_X_ED, fgj_X, fenvx, fenvy

# For Na, Vmem = 26mV*Z*ln(Cout/Cin), and Cin=Cout/exp(Vmem/(26mV*Z))
def Nernst(Vmem,c_env, Z):
    #print ('c_env=', c_env, ', Vmem=', Vmem, ', Z=', Z)
    return c_env / math.exp(Vmem / (.026*Z))