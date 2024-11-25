from qutip import *
import numpy as np


def solve_master_equation(n:int, omega_m:float, omega_c:float, gamma:list, kappa:list, g:float, t:list, initial_state:list, type:str) -> dict:
     
    """
    Solves the master equation for two molecules coupled to a cavity using QuTiP.

    Parameters
    ----------
    n : int
        Number of molecules.
    omega_m : list
        Transition frequencies of the molecules.
    omega_c : float
        Cavity mode frequency.
    gamma : list
        Decay constants of each molecule.
    kappa : list
        Cavity decay rates
    g : float
        Coupling strength.
    t : list
        Simulation time instants.
    initial_state : list
        Initial state of the molecules in the computational basis.
    type : str
        'original' for the original master equation with the cavity,
        'markovian' for the Markovian ME.

    Returns
    -------
    dict
        Dictionary containing the expectation values of the populations of the 
        computational and the coupled basis states (i.e., <$\\P_i$> where $\\P_i$ = |i><i|).
        The keys are '00', '01', '10', '11', 'G', '-', '+', 'E' and the values, a list
        with <$\\P_i> (t_i) for each t_i in `t`$

    Description
    -----------
        This function solves the master equation of a two TLSs, each with frequency `omega_m[i]`, 
        and decaying at a rate `gamma[i]`, while coupled each to a cavity through a coupling 
        strengths `g[i]`.


    """

    # This solver uses the mesolve function from QuTiP, which requires five 
    # inputs as of version 5.0.2:
    #
    # 1) H : QObj
    #       Hamiltonian.
    # 2) rho0 : Qobj
    #       Initial density matrix.
    # 3) tlist : list, array
    #       Time instants 
    # 4) c_ops : list of QobjEvo.
    #       Lindblad operators 
    # 5) e_ops : list of Qobj
    #       Observables to measure. 
    #
    # tlist and the initial state (rho0) are given as an input to this function and the rest are
    # calculated through other functions (defined below).

    system_dim = 2**n
    cavities_dim = 2**(n-1)

    # The total Hilbert space (tensor product space) has dimension
    # system_dim*cavities_dim = 2**(2n-1)

    total_dim = system_dim*cavities_dim

    hamiltonian = get_hamiltonian(n, omega_m, omega_c, kappa, g, type)

    initial_molecular_state = Qobj(np.array(initial_state)).unit()

    initial_cavity_state = basis(2**(n-1), 0)

    if type == 'original':

        rho0 = tensor(initial_molecular_state, initial_cavity_state)
        rho0.dims = [[total_dim],[1]]

    elif type == 'markovian':

        rho0 = initial_molecular_state
        rho0.dims = [[system_dim], [1]]

    observables = get_observables(n, type)

    lindblads = get_lindblads(n, omega_m, omega_c, gamma, kappa, g, type)

    result = mesolve(hamiltonian, rho0, t, lindblads, observables)

    evs = {}
 
    for key in observables.keys():

        evs[key] = result.expect[int(key)]
        
    return evs

def get_hamiltonian(n, omega_m, omega_c, kappa, g, type):

    """
    Returns the system's hamiltonian, including both H_0 and H_I

    Parameters
    ----------
    n : int
        Number of qubit in the system's register.
    omega_m : list
        Transition frequencies of the system.
    omega_c : float
        Cavity mode frequency
    kappa : list
        Cavity decay rates
    g : list
        Coupling strengths.
    type : str
        'original' for the original master equation with the cavity,
        'markovian' for the Markovian ME, 'diagonal' for the ME in the coupled
        basis with just the diagonal dissipators.

    Returns
    -------
    QObj
        System's hamiltonian

    """

    # Basis definition

    system_dim = 2**n
    cavities_dim = 2**(n-1)

    # The total Hilbert space (tensor product space) has dimension
    # system_dim*cavities_dim = 2**(2n-1)

    total_dim = system_dim*cavities_dim

    # One qubit basis

    ket_0 = Qobj([[1],[0]])

    ket_1 = Qobj([[0],[1]])

    # Operators definition

    sigma = [] # This will store all sigma_i operators
    a_ops = [] # This will store all a_i operators
    
    if type == 'original':

        sigmam = ket_0*ket_1.dag() 
        a = destroy(2)

        # We have to add extra identity operators to account
        # for the cavities 
        
        for i in range(n): 

            ops = [qeye(2)]*(2*n-1) # n molecules and n-1 cavities
            ops[i] = sigmam
            sig = tensor(ops)
            sig.dims = [[total_dim], [total_dim]]
            sigma.append(sig)
        
        # sigma = [tensor(x, qeye(2)) for x in sigma]

        for i in range(n-1):

            ops = [qeye(2)]*(2*n-1) # n molecules and n-1 cavities
            ops[i+n] = a
            aa = tensor(ops)
            aa.dims = [[total_dim], [total_dim]]
            a_ops.append(aa)

    elif type == 'markovian':

        sigmam = ket_0*ket_1.dag() 

        for i in range(n):

            ops = [qeye(2)]*(n) 
            ops[i] = sigmam
            sig = tensor(ops)
            sig.dims = [[system_dim], [system_dim]]
            sigma.append(sig)
   
    # Free qubit hamiltonian

    h = []

    if type == 'original':

        omega = omega_m
        g = g

        for i in range(n):

            h.append(omega[i]*sigma[i].dag()*sigma[i])
        
        for i in range(n-1):

            h.append(omega_c*a_ops[i].dag()*a_ops[i])

    elif type == 'markovian':

        delta = [ x - omega_c for x in omega_m ]
        # omega = [ omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]
        #g = [ (0.5*g[i]*g[i+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1) ]

        for i in range(n):

            if i==0:

                omega_eff = omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2)

            if i == (n-1):
                
                j = 2*i
                omega_eff = omega_m[i] + (delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2)
            
            else:   

                j = 2*i
                omega_eff = omega_m[i] + (delta[i]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + (delta[i]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2)

            h.append(omega_eff*sigma[i].dag()*sigma[i])
    
    h_0 = sum(h)

    # Pairwise interaction hamiltonian

    h = []

    if type == 'original':
    
        for i in range(n):

            if i==0:

                h.append(g[i]*(sigma[i].dag()*a_ops[i] + sigma[i]*a_ops[i].dag()))

            elif i == (n-1):

                j = 2*i
                h.append(g[j-1]*(sigma[i].dag()*a_ops[i-1] + sigma[i]*a_ops[i-1].dag()))

            else:

                j = 2*i
                h.append(g[j-1]*(sigma[i].dag()*a_ops[i-1] + sigma[i]*a_ops[i-1].dag()) + g[j]*(sigma[i].dag()*a_ops[i] + sigma[i]*a_ops[i].dag()))

            #h.append(g[i]*(sigma[i].dag()*a_ops[i] + sigma[i]*a_ops[i].dag()) + g[i+1]*(sigma[i+1].dag()*a_ops[i] + sigma[i+1]*a_ops[i].dag()))


    elif type == 'markovian':

        for i in range(0, n-1):

            #g_eff = (0.5*g[i]*g[i+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1])

            #if i>=1:        
            j = 2*i
            g_eff = (0.5*g[j]*g[j+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1])
                
            h.append(g_eff*(sigma[i].dag()*sigma[i+1] + sigma[i]*sigma[i+1].dag()))

    h_int = sum(h)

    hamiltonian = h_0 + h_int

    return hamiltonian

def get_observables(n:int, type:str):  

    """
    Returns a list with the observables being measured, 
    which in our case are the populations of the computational 
    basis states (i.e., <\Pi_i> where \Pi_i = |i><i|).

    Parameters
    ----------
    n : int
        Number of qubit in the system's register.
    type : str
        'original' for the original master equation with the cavity,
        'markovian' for the Markovian ME.

    Returns
    -------
    list[QObj]
        Observables

    """

    # Basis definition

    system_dim = 2**n
    cavities_dim = 2**(n-1)

    total_dim = system_dim*cavities_dim

    # One qubit basis

    ket_0 = Qobj([[1],[0]])

    ket_1 = Qobj([[0],[1]])

    # Operators definition

    observables = {} # This will store all sigma_i operators

    if type == 'original':

        sigmam = ket_0*ket_1.dag()
        sp_sm = sigmam.dag()*sigmam

        # We have to add extra identity operators to account
        # for the cavities 
        
        for i in range(n): 

            ops = [qeye(2)]*(2*n-1) # n molecules and n-1 cavities
            ops[i] = sp_sm
            sig = tensor(ops)
            sig.dims = [[total_dim], [total_dim]]
            observables[str(i)] = sig

    elif type == 'markovian':

        sigmam = ket_0*ket_1.dag() 
        sp_sm = sigmam.dag()*sigmam

        for i in range(n):

            ops = [qeye(2)]*(n) 
            ops[i] = sp_sm
            sig = tensor(ops)
            sig.dims = [[system_dim], [system_dim]]
            observables[str(i)] = sig

    #computational_basis = []
    #labels = []

    #for i in range(system_dim):

        #zeros = [0]*system_dim
        #zeros[i] = 1
        #computational_basis.append(Qobj(np.array(zeros).reshape(system_dim,1)))
        #labels.append(''.join(str(x) for x in zeros))

    # Projectors are defined and stored in a dictionary 

    #observables = {}
    #counter = 0

    #for ket in (computational_basis):

        #if type == 'original':

            #id = tensor([qeye(2)]*(n-1))

            #obs = tensor(ket*ket.dag(), id)

            #obs.dims = [[total_dim], [total_dim]]
        
            #observables[labels[counter]] = obs
        
        #elif type == 'markovian':

            #obs = ket*ket.dag()

            #obs.dims = [[system_dim], [system_dim]]

            #observables[labels[counter]] = obs

        #counter = counter + 1

    return observables


def get_lindblads(n, omega_m, omega_c, gamma, kappa, g, type):

    """
    Returns a list with the collapse operators.

    Parameters
    ----------
    n : int
        Number of molecules.
    omega_m : list
        Transition frequencies of the system.
    omega_c : float
        Cavity mode frequency
    gamma : list
        Sorted decay constants of each qubit.
    kappa : list
        Cavity decay rates
    g : list
        Coupling strengths.
    type : str
        'original' for the original master equation with the cavity,
        'markovian' for the Markovian ME.   

    Returns
    -------
    list[QObj]
        Collapse operators

    """

    # Basis definition

    system_dim = 2**n
    cavities_dim = 2**(n-1)

    total_dim = system_dim*cavities_dim

    # One qubit basis

    ket_0 = Qobj([[1],[0]])

    ket_1 = Qobj([[0],[1]])

    # Operators definition

    sigma = [] # This will store all sigma_i operators
    a_ops = [] # This will store all a_i operators

    # Operators definition

    if type == 'original':

        sigmam = ket_0*ket_1.dag() 
        a = destroy(2)

        # We have to add extra identity operators to account
        # for the cavities 
        
        for i in range(n): 

            ops = [qeye(2)]*(2*n-1) # n molecules and n-1 cavities
            ops[i] = sigmam
            sig = tensor(ops)
            sig.dims = [[total_dim], [total_dim]]
            sigma.append(sig)
        
        # sigma = [tensor(x, qeye(2)) for x in sigma]

        for i in range(n-1):

            ops = [qeye(2)]*(2*n-1) # n molecules and n-1 cavities
            ops[i+n] = a
            aa = tensor(ops)
            aa.dims = [[total_dim], [total_dim]]
            a_ops.append(aa)

        molecule_decay = [np.sqrt(gamma[i])*sigma[i] for i in range(n)]

        cavity_decay = [np.sqrt(kappa[0])*a_ops[i] for i in range(n-1)]

        lindblads = molecule_decay +  cavity_decay

    elif type == 'markovian':

        sigmam = ket_0*ket_1.dag() 

        for i in range(n):

            ops = [qeye(2)]*(n) 
            ops[i] = sigmam
            sig = tensor(ops)
            sig.dims = [[system_dim], [system_dim]]
            sigma.append(sig)

        delta = [ x - omega_c for x in omega_m ]

        molecule_effective_decay = []
        cross_decay_one = []
        cross_decay_two = []

        for i in range(0, n):

            if i == 0:

                eff_gamma = gamma[i] + (kappa[0]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2)

            elif i == (n-1):

                j = 2*i
                eff_gamma = gamma[i] + (kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2)

            else:
                
                j = 2*i
                eff_gamma = gamma[i] + (kappa[0]*(g[j-1]**2))/((0.5*kappa[0])**2 + delta[i]**2) + (kappa[0]*(g[j]**2))/((0.5*kappa[0])**2 + delta[i]**2)

            molecule_effective_decay.append(np.sqrt(eff_gamma)*sigma[i])

        for i in range(n-1):

            j = 2*i
            gamma_cross = (g[j]*g[j+1]*(kappa[0]))/((kappa[0]/2)**2 + delta[i]*delta[i+1])

            cross_decay_one.append(gamma_cross*lindblad_dissipator(a = sigma[i], b = sigma[i+1]))

            cross_decay_two.append(gamma_cross*lindblad_dissipator(a = sigma[i+1], b = sigma[i]))
        
        lindblads = molecule_effective_decay + cross_decay_one + cross_decay_two

    return lindblads 