from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate, ControlledGate
import numpy as np
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.library import RYGate, RYYGate, Initialize


def create_markovian_circuit(n:int, omega_m:list, omega_c:list, g:list, gamma:list, kappa:list, initial_state:list, r:int, type:str) -> QuantumCircuit:

    """
    Creates the quantum circuit representing the markovian master equation
    in the coupled basis in terms of a parametric t.

    Parameters
    ----------
    n : int
        Number of qubits in the system's register (i.e., number of molecules).
    omega_m : list
        Transition frequencies of the molecules.
    omega_c : float
        Cavity mode frequency.
    g : float
        Coupling strength.
    gamma : list
        Decay constants of each molecule.
    kappa : list
        Cavity decay rates
    initial_state : list
        Initial state of the molecules in the computational basis.
    r : int
        Number of core circuit repetitions (Trotter steps)
    type : str
        'regular' if only D is in the coupled basis and 'diagonal'
        if everything is in the coupled basis.


    Returns
    -------
    QuantumCircuit
        The output parametrized circuit, t-dependent.

    Description
    -----------
    

    Examples
    --------

    """

    # System parameters:

    delta = [ x - omega_c for x in omega_m ]
    
    omega_eff = [ omega_m[i] + (delta[i]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]
    
    g_eff = [ (0.5*g[i]*g[i+1]*(delta[i] + delta[i+1]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1) ]

    if type == 'diagonal':

        omega_eff[0] = omega_eff[0] + g_eff[0]

        omega_eff[1] = omega_eff[1] - g_eff[0]

    gamma_eff = [ gamma[i] + (kappa[0]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]

    gamma_cross = [ (g[i]*g[i+1]*(kappa[0]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1)]  
    
    gamma_plus = (gamma_eff[0] + gamma_eff[1])/2 + gamma_cross[0]

    gamma_minus = (gamma_eff[0] + gamma_eff[1])/2 - gamma_cross[0]

    # Register creation

    system = QuantumRegister(n, 'q')
    environment = QuantumRegister(n, 'e')

    # Parameter definition

    delta_t = Parameter('$t$')/r

    beta = [ 2*omega_eff[j]*delta_t for j in range(n) ]

    alpha = [ g_eff[j]*delta_t for j in range(n-1) ]

    theta_plus = (((1 - (-delta_t*gamma_plus).exp())**(1/2)).arcsin())*2

    theta_minus = (((1 - (-delta_t*gamma_minus).exp())**(1/2)).arcsin())*2

    #theta_plus = ((delta_t*gamma_plus)**(1/2))*2

    #theta_minus = ((delta_t*gamma_minus)**(1/2))*2

    theta = [theta_minus, theta_plus]
    
    # Initialization circuit

    init = QuantumCircuit(system, environment)

    if type == 'regular':

        initial_qubit_state = initial_state    
    
    elif type == 'diagonal':

        P = np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]])

        initial_qubit_state = np.matmul(P, initial_state)

    initial_statevector = Statevector(initial_qubit_state) 

    init.initialize(params = initial_statevector, qubits = system, normalize = True)

    # Free Hamiltonian evolution

    u_0 = QuantumCircuit(system, environment)

    for i in range(n):

        u_0.rz(beta[i], system[i])

    # Two-layer interaction circuit

    if type == 'regular':

        u1 = QuantumCircuit(system, environment)

        for j in range(0, n-1, 2):

            u1.ryy(alpha[j], qubit1 = system[j], qubit2 = system[j+1])
            u1.rxx(alpha[j], qubit1 = system[j], qubit2 = system[j+1])

        if n > 2:

            u2 = QuantumCircuit(system, environment)

            for k in range(1, n-1, 2):

                u2.ryy(alpha[k], qubit1 = system[k], qubit2 = system[k+1])
                u2.rxx(alpha[k], qubit1 = system[k], qubit2 = system[k+1])

            u_int = u1.compose(u2)

        else:

            u_int = u1

    # Decay layer

    u_decay = QuantumCircuit(system, environment)

    if type == 'regular':

        P = np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]])

        P_gate = Operator(P).to_instruction()

        P_dag_gate = Operator(P.transpose()).to_instruction()

        u_decay.append(P_gate, system)

    ccry_plus = RYGate(theta_plus - theta_minus).control(2)

    ccry_minus = RYGate(theta_minus - theta_plus).control(2)

    u_decay.append(ccry_plus, [system[0], system[1], environment[0]])

    u_decay.cry(theta[0], system[0], environment[0])

    u_decay.append(ccry_minus, [system[0], system[1], environment[1]])

    u_decay.cry(theta[1], system[1], environment[1])

    u_decay.cx(environment[0], system[0])

    u_decay.cx(environment[1], system[1])

    if type == 'regular':

        u_decay.append(P_dag_gate, system)

    u_decay.reset(environment)

    # Putting everything (except the initialization) together
    # for Trotterization

    if type == 'regular':

        u = ((u_0.compose(u_int)).compose(u_decay))

    elif type == 'diagonal':

        u = u_0.compose(u_decay)

    trotterized_u = u.repeat(r).decompose()

    # Finally, we put the initialization right at the beginning.
    
    parametrized_qc = trotterized_u.compose(init, front = True)

    return parametrized_qc

def get_circuit_properties(qc:QuantumCircuit) -> dict:

    """
    Retrieves the main properties of a given circuit

    Parameters
    ----------
    qc : QuantumCircuit
    Circuit whose properties are wished to be retrieved.

    Returns
    -------
    dict
    Dictionary with available keys 'width', 'size', 'two qubit gates',
    'depth' and 'two qubit depth'.

    Description
    -----------

    """

    properties = {}

    properties['width'] = qc.width()
    properties['size'] = qc.size()
    properties['two qubit gates'] = qc.num_nonlocal_gates()
    properties['depth'] = qc.depth()
    properties['two qubit depth'] = qc.depth(lambda instr: len(instr.qubits) > 1)

    return properties