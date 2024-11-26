from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp, Operator

from typing import Union
from qiskit.providers import BackendV1, BackendV2

from main_functions.efficient_circuit_markovian_WORKS import create_markovian_circuit
import numpy as np

# This is an upgraded version of the qiskit_solver function found in qiskit_solver.py. It needs
# to make one call to the Estimator to compute the expectation values of the observables for any time 
# array.

def markovian_qiskit_solver(n:int, omega_m:list, omega_c:float, gamma:list, g:float, kappa:list, t:list, r:int, initial_state:list,
                  backend:Union[str, BackendV1, BackendV2], optimization_level:int, options:dict, type:str) -> dict:

    """
    Computes the time evolution of the population of all basis states (computational and coupled)
    under free evolution, decay and pairwise interaction.
    
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
    t : list
        Simulation time instants.
    backend : Union[str, BackendV1, BackendV2]
        Source of the default values for the basis_gates, inst_map, coupling_map, 
        backend_properties, instruction_durations, timing_constraints, and target
    optimization_level : int
        Level of optimization to generate the PassManager
        (see https://docs.quantum.ibm.com/guides/set-optimization for more info)
    options : dict
        Dictionary containing the specific EstimatorV2 options
        (see https://docs.quantum.ibm.com/guides/runtime-options-overview)


    Returns
    -------
    list
        A list of lists where list[i] corresponds to the expectation values of observable[i]
        computed on the transpiled circuit evaluated at each parameter value fed to the Estimator. 
        In our case, list[i] represents the evolution of the excited state population of qubit i.

    Description
    -----------
    
    Examples

    """
     
    qc = create_markovian_circuit(n = n, omega_m = omega_m, omega_c = omega_c,
                                    g = g, gamma = gamma, kappa = kappa,
                                    initial_state = initial_state, r = r, type = type)
            
    evs = efficient_ev_calculator(n = n,
                                       qc = qc,
                                       t = t,
                                       backend = backend,
                                       optimization_level = optimization_level,
                                       options = options, type = type)
    
    return evs



def efficient_ev_calculator(n:int, qc:QuantumCircuit, t:list, backend:Union[str, BackendV1, BackendV2], 
                  optimization_level:int, options:dict, type:str) -> dict:
    

    # n = int(qc.num_qubits*0.5)

    pm = generate_preset_pass_manager(backend  = backend, optimization_level = optimization_level)
    transpiled_qc = pm.run(qc)

    observables, labels = get_observables(type = type)
    transpiled_observables = [x.apply_layout(transpiled_qc.layout) for x in observables.values()]

    reshaped_obs = np.fromiter(transpiled_observables, dtype=object)
    reshaped_obs = reshaped_obs.reshape((len(observables), 1))

    instants = [[x] for x in t]

    estimator = EstimatorV2(mode = backend, options = options)
    job = estimator.run([(transpiled_qc, reshaped_obs, instants)])

    evs = {}

    counter = 0
 
    for l in labels:

        evs[l] = job.result()[0].data.evs[counter]
        counter = counter + 1

    return evs


def get_observables(type:str):

    """
    Returns a list with the observables being measured, 
    which in our case are the populations of the computational and 
    the coupled basis states (i.e., <\Pi_i> where \Pi_i = |i><i|).

    Parameters
    ----------

    type : str
        'regular' if only D is in the coupled basis and 'diagonal'
        if everything is in the coupled basis.
    
    Returns
    -------
    list[SparsePauliOp]
        Observables

    """

    # The observables must be SparsePauliOps, so we are going to create them
    # using the from_operator method and from the matrix representation of the
    # projectors.

    # Basis definition

    # Computational basis 

    ket_0 = np.array([[1],[0]])
    ket_1 = np.array([[0],[1]])

    ket_00 = np.kron(ket_0,ket_0)
    ket_01 = np.kron(ket_0, ket_1) 
    ket_10 = np.kron(ket_1, ket_0) 
    ket_11 = np.kron(ket_1, ket_1)

    computational_basis = [ket_00, ket_01, ket_10, ket_11]

    # Coupled basis

    ket_G = ket_00
    ket_lambda_minus = 1/(np.sqrt(2))*( -ket_01 + ket_10 )
    ket_lambda_plus = 1/(np.sqrt(2))*( ket_01 + ket_10 )
    ket_E = ket_11

    coupled_basis = [ket_G, ket_lambda_minus, ket_lambda_plus, ket_E]

    # Projectors are defined and stored in a dictionary 

    observables = {}
    labels = ['00', '01', '10', '11', 'G', '-', '+', 'E']

    counter = 0

    P = np.array([[1, 0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 0, 1]])

    for ket in (computational_basis + coupled_basis):

        if type == 'regular':

            matrix_rep = ket*ket.transpose()
        
        elif type == 'diagonal':

            matrix_rep = np.matmul(np.matmul(P.transpose(), ket*ket.transpose()), P).round(2)
        
        observables[labels[counter]] = SparsePauliOp('II').tensor(SparsePauliOp.from_operator(matrix_rep))

        # Identity string to account for the ancilla register.

        counter = counter + 1

    return observables, labels