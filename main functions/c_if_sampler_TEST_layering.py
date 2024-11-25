# Implement two types: with n ancillas and with 1 ancilla per pair.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RYGate
import numpy as np
from qiskit.quantum_info import Operator, Statevector
import math
from qiskit.transpiler import Layout


def get_circuit_sampler_layered(n:int, omega_m:list, omega_c:list, g:list, gamma:list, kappa:list, initial_state:list, r:int, type:str) -> QuantumCircuit:

    """


    Returns
    -------


    Description
    -----------
    'one ancilla' : one ancilla per pair
    'regular' : n ancillas

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

    if type == 'one ancilla':

        environment = QuantumRegister(math.trunc(n/2), 'e')
    else:

        environment = QuantumRegister(n, 'e')
        
    classical_bits = ClassicalRegister(n-1, 'c')

    # Parameter definition

    delta_t = Parameter('$t$')/r
    beta = [ 2*omega_eff[j]*delta_t for j in range(n) ]
    alpha = [ g_eff[j]*delta_t for j in range(n-1) ]
    theta_plus = [ (((1 - (-delta_t*gamma_plus[j]).exp())**(1/2)).arcsin())*2 for j in range(n-1)]
    theta_minus = [ (((1 - (-delta_t*gamma_minus[j]).exp())**(1/2)).arcsin())*2 for j in range(n-1)]

    # Initialization circuit

    init = QuantumCircuit(system, environment, classical_bits)
    initial_qubit_state = initial_state    
    initial_statevector = Statevector(initial_qubit_state) 
    init.initialize(params = initial_statevector, qubits = system, normalize = True)

    qc = QuantumCircuit(system, environment, classical_bits)

    # Free Hamiltonian evolution
    for l in range(r):

        for i in range(n):

            qc.rz(beta[i], system[i])

        # Two-layer interaction circuit

        P = np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]])

        P_gate = Operator(P).to_instruction()

        P_dag_gate = Operator(P.transpose()).to_instruction()

        counter = 0

        for j in range(0, n-1, 2):

            qc.ryy(alpha[j], qubit1 = system[j], qubit2 = system[j+1])
            qc.rxx(alpha[j], qubit1 = system[j], qubit2 = system[j+1])

            if type == 'one ancilla':

                qc.append(P_gate, [system[j], system[j+1]])

                qc.ccx(system[j], system[j+1], environment[counter])

                qc.measure(environment[counter], classical_bits[j])

                # Checking classically for correct parameter assignment

                with qc.if_test((classical_bits[j], 0)) as else_:

                    qc.cry(theta_minus[j], system[j], environment[counter])
                    qc.cx(environment[counter], system[j])
                    qc.reset(environment[counter])
                    qc.cry(theta_plus[j], system[j+1], environment[counter])               
                    qc.cx(environment[counter], system[j+1])

                with else_:

                    qc.x(environment[counter]) # To make it go back to 1.
                    qc.cry(theta_plus[j], system[j], environment[counter])
                    qc.cx(environment[counter], system[j])
                    qc.reset(environment[counter])
                    qc.cry(theta_minus[j], system[j+1], environment[counter])               
                    qc.cx(environment[counter], system[j+1])


                qc.append(P_dag_gate, [system[j], system[j+1]])

                qc.reset(environment[counter])

                counter = counter + 1

            else:           

                qc.append(P_gate, [system[j], system[j+1]])

                qc.ccx(system[j], system[j+1], environment[j])

                qc.measure(environment[j], classical_bits[j])

                # Checking classically for correct parameter assignment

                with qc.if_test((classical_bits[j], 0)) as else_:

                    qc.cry(theta_minus[j], system[j], environment[j])
                    qc.cry(theta_plus[j], system[j+1], environment[j+1])
                    qc.cx(environment[j], system[j])
                    qc.cx(environment[j+1], system[j+1])

                with else_:

                    qc.x(environment[j]) # To make it go back to 1.
                    qc.cry(theta_plus[j], system[j], environment[j])
                    qc.cry(theta_minus[j], system[j+1], environment[j+1])
                    qc.cx(environment[j], system[j])
                    qc.cx(environment[j+1], system[j+1])

                qc.append(P_dag_gate, [system[j], system[j+1]])

                qc.reset([environment[j], environment[j+1]])

        if n > 2:

            #u2 = QuantumCircuit(system, environment)

            counter = 0

            for j in range(1, n-1, 2):

                qc.ryy(alpha[j], qubit1 = system[j], qubit2 = system[j+1])
                qc.rxx(alpha[j], qubit1 = system[j], qubit2 = system[j+1])

                if type == 'one ancilla':

                    qc.append(P_gate, [system[j], system[j+1]])

                    qc.ccx(system[j], system[j+1], environment[counter])

                    qc.measure(environment[counter], classical_bits[j])

                    # Checking classically for correct parameter assignment

                    with qc.if_test((classical_bits[j], 0)) as else_:

                        qc.cry(theta_minus[j], system[j], environment[counter])
                        qc.cx(environment[counter], system[j])
                        qc.reset(environment[counter])
                        qc.cry(theta_plus[j], system[j+1], environment[counter])               
                        qc.cx(environment[counter], system[j+1])

                    with else_:

                        qc.x(environment[counter]) # To make it go back to 1.
                        qc.cry(theta_plus[j], system[j], environment[counter])
                        qc.cx(environment[counter], system[j])
                        qc.reset(environment[counter])
                        qc.cry(theta_minus[j], system[j+1], environment[counter])               
                        qc.cx(environment[counter], system[j+1])


                    qc.append(P_dag_gate, [system[j], system[j+1]])

                    qc.reset(environment[counter])

                    counter = counter + 1
                
                else:         
                    
                    qc.append(P_gate, [system[j], system[j+1]])

                    qc.ccx(system[j], system[j+1], environment[j])

                    qc.measure(environment[j], classical_bits[j])

                    # Checking classically for correct parameter assignment

                    with qc.if_test((classical_bits[j], 0)) as else_:

                        qc.cry(theta_minus[j], system[j], environment[j])
                        qc.cry(theta_plus[j], system[j+1], environment[j+1])
                        qc.cx(environment[j], system[j])
                        qc.cx(environment[j+1], system[j+1])

                    with else_:
                        
                        qc.x(environment[j])
                        qc.cry(theta_plus[j], system[j], environment[j])
                        qc.cry(theta_minus[j], system[j+1], environment[j+1])
                        qc.cx(environment[j], system[j])
                        qc.cx(environment[j+1], system[j+1])

                    qc.append(P_dag_gate, [system[j], system[j+1]])

                    qc.reset([environment[j], environment[j+1]])


    # Finally, we put the initialization right at the beginning and 
    # add the measurements for the Sampler implementation
    
    parametrized_qc = (qc.compose(init, front = True))
    parametrized_qc.measure_all()

    # Getting the custom layout for the one ancilla case

    init_layout = custom_init_layout(n, system, environment)

    return parametrized_qc, init_layout

def custom_init_layout(n, system, environment):

    physical_qubits = list(range(16)) + [19] + list(range(35,20,1))
    init_layout = Layout()
    
    site = 0

    for i in range(0,n,2):

        init_layout.add(system[i], physical_qubits[site])
        site = site + 1

        if i+1<=(n-1):

            init_layout.add(system[i+1], physical_qubits[site])
            site = site + 1

        j = int(i/2)

        if j<(math.trunc(n/2)):

            init_layout.add(environment[j], physical_qubits[site])

        site = site +1
    
    return init_layout