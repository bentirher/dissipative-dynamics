from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate, ControlledGate
import numpy as np
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.library import RYGate, RYYGate, Initialize


def create_markovian_circuit(n:int, omega_m:list, omega_c:list, g:list, gamma:list, kappa:list, initial_state:list, r:int) -> QuantumCircuit:

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

    gamma_eff = [ gamma[i] + (kappa[0]*(g[i]**2))/((0.5*kappa[0])**2 + delta[i]**2) for i in range(n) ]

    gamma_cross = [ (g[i]*g[i+1]*(kappa[0]))/((kappa[0]/2)**2 + delta[i]*delta[i+1]) for i in range(n-1)]  
    
    gamma_plus = [(gamma_eff[i] + gamma_eff[i+1])/2 + gamma_cross[i] for i in range(n-1)]

    gamma_minus = [(gamma_eff[i] + gamma_eff[i+1])/2 - gamma_cross[i] for i in range(n-1)]

    # Register creation

    system = QuantumRegister(n, 'q')
    environment = QuantumRegister(n, 'e')

    # Parameter definition

    delta_t = Parameter('$t$')/r

    beta = [ 2*omega_eff[j]*delta_t for j in range(n) ]

    alpha = [ g_eff[j]*delta_t for j in range(n-1) ]

    theta_plus = [ (((1 - (-delta_t*gamma_plus[j]).exp())**(1/2)).arcsin())*2 for j in range(n-1)]

    theta_minus = [ (((1 - (-delta_t*gamma_minus[j]).exp())**(1/2)).arcsin())*2 for j in range(n-1)]

    #theta_plus = ((delta_t*gamma_plus)**(1/2))*2

    #theta_minus = ((delta_t*gamma_minus)**(1/2))*2
    
    # Initialization circuit

    init = QuantumCircuit(system, environment)
    
    initial_qubit_state = initial_state    

    initial_statevector = Statevector(initial_qubit_state) 

    init.initialize(params = initial_statevector, qubits = system, normalize = True)

    # Free Hamiltonian evolution

    u_0 = QuantumCircuit(system, environment)

    for i in range(n):

        u_0.rz(beta[i], system[i])

    # Two-layer interaction circuit

    
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

    u_decay_first_layer = QuantumCircuit(system, environment)

    # Basis change gate definition

    P = np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]])

    P_gate = Operator(P).to_instruction()

    P_dag_gate = Operator(P.transpose()).to_instruction()

    # Now we apply P to the first layer, as in the interaction circuit and the decay part.

    for j in range(0, n-1, 2):

        u_decay_first_layer.append(P_gate, [system[j], system[j+1]])

        ccry_plus = RYGate(theta_plus[j] - theta_minus[j]).control(2)

        ccry_minus = RYGate(theta_minus[j] - theta_plus[j]).control(2)

        u_decay_first_layer.append(ccry_plus, [system[j], system[j+1], environment[j]])

        u_decay_first_layer.append(ccry_minus, [system[j], system[j+1], environment[j+1]])

        u_decay_first_layer.cry(theta_minus[j], system[j], environment[j])

        u_decay_first_layer.cry(theta_plus[j], system[j+1], environment[j+1])

        u_decay_first_layer.cx(environment[j], system[j])

        u_decay_first_layer.cx(environment[j+1], system[j+1])

        u_decay_first_layer.append(P_dag_gate, [system[j], system[j+1]])

        u_decay_first_layer.reset([environment[j], environment[j+1]])
    
    # And onto the second layer

    if n > 2:

        u_decay_second_layer = QuantumCircuit(system, environment)

        for j in range(1, n-1, 2):

            u_decay_first_layer.append(P_gate, [system[j], system[j+1]])

            ccry_plus = RYGate(theta_plus[j] - theta_minus[j]).control(2)

            ccry_minus = RYGate(theta_minus[j] - theta_plus[j]).control(2)

            u_decay_second_layer.append(ccry_plus, [system[j], system[j+1], environment[j]])

            u_decay_second_layer.append(ccry_minus, [system[j], system[j+1], environment[j+1]])

            u_decay_second_layer.cry(theta_minus[j], system[j], environment[j])

            u_decay_second_layer.cry(theta_plus[j], system[j+1], environment[j+1])

            u_decay_second_layer.cx(environment[j], system[j])

            u_decay_second_layer.cx(environment[j+1], system[j+1])

            u_decay_second_layer.append(P_dag_gate, [system[j], system[j+1]])

            u_decay_second_layer.reset([environment[j], environment[j+1]])

        u_decay = u_decay_first_layer.compose(u_decay_second_layer)

    else:

        u_decay = u_decay_first_layer


    # Putting everything (except the initialization) together
    # for Trotterization

    u = ((u_0.compose(u_int)).compose(u_decay))

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