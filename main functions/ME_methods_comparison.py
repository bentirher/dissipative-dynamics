from qutip import *
import numpy as np


def solve_master_equation(omega_m:float, omega_c:float, gamma:list, kappa:list, g:float, t:list, initial_state:list, type:str) -> dict:
     
    """
    Solves the master equation for two molecules coupled to a cavity using QuTiP.

    Parameters
    ----------
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
        'markovian' for the Markovian ME, 
        'diagonal' for the ME in the coupled basis with just the diagonal dissipators.

    Returns
    -------
    dict
        Dictionary containing the expectation values of the populations of the 
        computational and the coupled basis states (i.e., <$\\P_i$> where $\\P_i$ = |i><i|).
        The keys are '00', '01', '10', '11', 'G', '-', '+', 'E' and the values, a list
        with <$\\P_i> (t_i) for each t_i in `t`$

    Description
    -----------
        This function solves the master equation for two TLSs, each with frequency `omega_m[i]`, 
        and decaying at a rate `gamma[i]`, while coupled to a cavity through a coupling 
        constant `g[i]`.


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

    n = 2

    hamiltonian = get_hamiltonian(n, omega_m, omega_c, kappa, g, type)

    initial_molecular_state = Qobj(initial_state)

    if type == 'original':

        initial_molecular_state.dims = [[2,2],[1,1]]

        rho0 = tensor(initial_molecular_state, basis(2,0))

    elif type == 'markovian':

        initial_molecular_state.dims = [[2,2],[1,1]]

        rho0 = initial_molecular_state

    elif type == 'diagonal':
        
        P = Qobj(np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]]))

        # P.dims = [[2,2],[2,2]]

        rho0 = P*initial_molecular_state

    observables, labels = get_observables(type)

    lindblads = get_lindblads(n, omega_m, omega_c, gamma, kappa, g, type)

    result = mesolve(hamiltonian, rho0, t, lindblads, observables)

    evs = {}

    counter = 0
 
    for l in labels:

        evs[l] = result.expect[counter]
        counter = counter + 1

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
        'markovian' for the Markovian ME, 
        'diagonal' for the ME in the coupled basis with just the diagonal dissipators.

    Returns
    -------
    QObj
        System's hamiltonian

    """

    # Basis definition

    if type == 'diagonal':

        gr = Qobj([[1],[0],[0],[0]])
        lambda_minus = Qobj([[0],[1],[0],[0]])
        lambda_plus = Qobj([[0],[0],[1],[0]])
        e = Qobj([[0],[0],[0],[1]])
    
    else:

        ket_0 = Qobj([[1],[0]])
        ket_1 = Qobj([[0],[1]])

    # Operators definition

    sigma = [] # This will store all sigma_i operators
    
    if type == 'original':

        sigmam = ket_0*ket_1.dag() 

        # We have to add an extra identity operator to account
        # for the cavity (an extra N-dimensional space)
        
        for i in range(n): 

            ops = [qeye(2)]*(n+1) # n molecules and 1 cavity
            ops[i] = sigmam
            sigma.append(tensor(ops))
        
        # sigma = [tensor(x, qeye(2)) for x in sigma]
        a = tensor(qeye(2), qeye(2), destroy(2))

    elif type == 'markovian':

        sigmam = ket_0*ket_1.dag() 

        for i in range(n):

            ops = [qeye(2)]*(n) 
            ops[i] = sigmam
            sigma.append(tensor(ops))
   
    # Free qubit hamiltonian

    h = []

    if type == 'original':

        omega = omega_m
        g = g

        for i in range(n):

            h.append(omega[i]*sigma[i].dag()*sigma[i])

        h.append(omega_c*a.dag()*a)

    elif type == 'markovian':

        delta = [ x - omega_c for x in omega_m ]
        omega = [ omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]
        g = [ (0.5*g[i]*g[i+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1) ]

        for i in range(n):

            h.append(omega[i]*sigma[i].dag()*sigma[i])
    
    h_0 = sum(h)

    # Pairwise interaction hamiltonian

    h = []

    if type == 'original':
    
        for i in range(n):

            h.append(g[i]*(sigma[i].dag()*a + sigma[i]*a.dag()))

    elif type == 'markovian':

        for i in range(n-1):

            h.append(g[i]*(sigma[i].dag()*sigma[i+1] + sigma[i]*sigma[i+1].dag()))

    h_int = sum(h)

    hamiltonian = h_0 + h_int

    # Diagonal hamiltonian

    if type == 'diagonal':

        delta = [ x - omega_c for x in omega_m ]
        omega = [ omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]
        g = [ (0.5*g[i]*g[i+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1) ]

        hamiltonian = 0*gr*gr.dag() + (omega[0] - g[0])*lambda_minus*lambda_minus.dag() + (omega[0] + g[0])*lambda_plus*lambda_plus.dag() + (2*omega[0])*e*e.dag()

    return hamiltonian

def get_observables(type):  ## NOT DEFINED FOR THE N QUBIT CASE

    """
    Returns a list with the observables being measured, 
    which in our case are the populations of the computational and 
    the coupled basis states (i.e., <\Pi_i> where \Pi_i = |i><i|).

    Parameters
    ----------
    type : str
        'original' for the original master equation with the cavity,
        'markovian' for the Markovian ME,
        'diagonal' for the ME in the coupled basis with just the diagonal dissipators.

    Returns
    -------
    list[QObj]
        Observables

    """

    # Basis definition

    # Computational basis 

    ket_0 = Qobj([[1],[0]])
    ket_1 = Qobj([[0],[1]])

    ket_00 = tensor(ket_0,ket_0)
    ket_01 = tensor(ket_0, ket_1)
    ket_10 = tensor(ket_1, ket_0)
    ket_11 = tensor(ket_1, ket_1)

    computational_basis = [ket_00, ket_01, ket_10, ket_11]

    # Coupled basis

    ket_G = ket_00
    ket_lambda_minus = 1/(np.sqrt(2))*(- ket_01 + ket_10 )
    ket_lambda_plus = 1/(np.sqrt(2))*( ket_01 + ket_10 )
    ket_E = ket_11

    coupled_basis = [ket_G, ket_lambda_minus, ket_lambda_plus, ket_E]

    # Projectors are defined and stored in a dictionary 

    observables = {}
    labels = ['00', '01', '10', '11', 'G', '-', '+', 'E']

    counter = 0

    for ket in (computational_basis + coupled_basis):

        if type == 'original':
        
            observables[labels[counter]] = tensor(ket*ket.dag(), qeye(2))
        
        elif type == 'markovian':

            observables[labels[counter]] = ket*ket.dag()

        elif type == 'diagonal':

            P = Qobj(np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]]))

            #P.dims = [[2,2],[2,2]]

            ket.dims = [[4],[1]]

            observables[labels[counter]] = (P.dag())*(ket*ket.dag())*P

        counter = counter + 1

    return observables, labels


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
        'markovian' for the Markovian ME,
        'diagonal' for the ME in the coupled basis with just the diagonal dissipators.  

    Returns
    -------
    list[QObj]
        Collapse operators

    """

    # Basis definition

    if type == 'diagonal':

        gr = Qobj([[1],[0],[0],[0]])
        lambda_minus = Qobj([[0],[1],[0],[0]])
        lambda_plus = Qobj([[0],[0],[1],[0]])
        e = Qobj([[0],[0],[0],[1]])

    else:

        ket_0 = Qobj([[1],[0]])
        ket_1 = Qobj([[0],[1]])

    # Operators definition

    if type == 'diagonal':

        sigma_gplus = Qobj(gr*lambda_plus.dag())
        sigma_gminus = Qobj(gr*lambda_minus.dag())
        sigma_pluse = Qobj(lambda_plus*e.dag())
        sigma_minuse = Qobj(lambda_minus*e.dag())
    
    else:

        sigmam = ket_0*ket_1.dag()      

        sigma = [] # This will store all sigma_i operators

        for i in range(n): 

            ops = [qeye(2)]*(n) # n molecules and 1 cavity
            ops[i] = sigmam
            sigma.append(tensor(ops))
    
    if type == 'original':

        # We have to add an extra identity operator to account
        # for the cavity (an extra N-dimensional space)
        sigma = [tensor(x, qeye(2)) for x in sigma]
        a = tensor(qeye(2), qeye(2), destroy(2))

        individual_decay = [np.sqrt(gamma[i])*sigma[i] for i in range(n)]

        individual_decay.append(np.sqrt(kappa[0])*a)

        lindblads = individual_decay 

    else:

        delta = [ x - omega_c for x in omega_m ]

        gamma = [ gamma[i] + (kappa[0]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]

        gamma_cross = [ (g[i]*g[i+1]*(kappa[0]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1)]

        if type == 'markovian': 

            individual_decay = [np.sqrt(gamma[i])*sigma[i] for i in range(n)]

            cross_decay = [gamma_cross[0]*lindblad_dissipator(a = sigma[0], b = sigma[1]), gamma_cross[0]*lindblad_dissipator(a = sigma[1], b = sigma[0])]
        
        elif type == 'diagonal':

            gamma_plus = (gamma[0] + gamma[1])/2 + gamma_cross[0]

            gamma_minus = (gamma[0] + gamma[1])/2 - gamma_cross[0] 

            individual_decay = [np.sqrt(gamma_plus)*sigma_gplus, np.sqrt(gamma_minus)*sigma_gminus, np.sqrt(gamma_plus)*sigma_pluse,  np.sqrt(gamma_minus)*sigma_minuse]

            cross_decay = []

        lindblads = individual_decay + cross_decay

    return lindblads 