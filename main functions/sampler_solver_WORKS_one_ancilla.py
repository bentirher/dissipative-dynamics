#from c_if_sampler_TEST_layering import get_circuit_sampler_layered as get_circuit_sampler
from c_if_sampler_WORKS_one_ancilla import get_circuit_sampler

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime import SamplerV2, Batch, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp, Operator

from typing import Union
from qiskit.providers import BackendV1, BackendV2

import numpy as np
import math
from qiskit.transpiler import Layout


def sampler_solver(n:int, omega_m:list, omega_c:float, gamma:list, g:float, kappa:list, t:list, r:int, initial_state:list,
                  backend:Union[str, BackendV1, BackendV2], optimization_level:int, options:dict, type:str) -> dict:
    
    qc = get_circuit_sampler(n, omega_m, omega_c, g, gamma, kappa, initial_state, r, type)

    #service = QiskitRuntimeService(
        #channel='ibm_quantum',
        #instance='ibm-q-ikerbasque/upv-ehu/dynamics-of-mole',
        #token='ac55769048d74690dcec2e0219671ebcfb53eb44d32ee50608858b25572950cf0c789f133deb5ff884e342bffa02bed17167a747733f4e6ea69fd531bf4f39d7'
    #)
    #session = Session(service = service, backend = backend)
    #sampler = SamplerV2(session = session, backend = backend, options = options)
    
    shots = options['default_shots']

    evs = {}

    for i in range(n):

        evs[str(i)] = []

    #pm = generate_preset_pass_manager(backend = backend, optimization_level = optimization_level, initial_layout = init_layout)
    pm = generate_preset_pass_manager(backend = backend, optimization_level = optimization_level)
    trans_qc = pm.run(qc)
    pubs = [(trans_qc, x) for x in t]

    ############ EXPERIMENTAL ##############

    max_circuits = 49

    jobs = []
    
    with Batch(backend=backend):

        sampler = SamplerV2(options = options)

        for i in range(0, len(t), max_circuits):

            if i + max_circuits <= len(t):

                job = sampler.run(pubs[i : i + max_circuits])

            else:
                
                job = sampler.run(pubs[i : len(t)])

            jobs.append(job)

    # Classical post-processing
    
    for j in jobs: 

        result = j.result() 

        # All jobs (except the last one typically) will contain the same number of PUBs so we have to iterate over the number of PUBs
        # which is equal to len(result)

        for k in range(len(result)):

            pub_result = result[k]
            counts = pub_result.data.meas.get_counts()
            #evs = get_spsm_evs(n, counts, shots, evs)
            states = [key[n:] for key in counts.keys()] # Output states
        
            if type == 'one ancilla':

                states = [key[math.trunc(n/2):] for key in counts.keys()]

            coeff = [ np.sqrt(counts[key]/shots) for key in counts.keys()] # Normalized coefficients
            eigenvalues = [1, -1] # Z eigenvalues

            for i in range(n):

                evs[str(i)].append(0.5*( 1 - sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))) # Fix this
    

    ########### EXPERIMENTAL ###############

    ########## THIS WORKS ##########

    

    # with Batch(backend=backend):

    #     sampler = SamplerV2(options = options)
    #     job = sampler.run(pubs)

    # for k in range(len(t)):

    #     pub_result = job.result()[k]
    #     counts = pub_result.data.meas.get_counts()
    #     #evs = get_spsm_evs(n, counts, shots, evs)
    #     states = [key[n:] for key in counts.keys()] # Output states
        
    #     if type == 'one ancilla':

    #         states = [key[math.trunc(n/2):] for key in counts.keys()]

    #     coeff = [ np.sqrt(counts[key]/shots) for key in counts.keys()] # Normalized coefficients
    #     eigenvalues = [1, -1] # Z eigenvalues

    #     for i in range(n):

    #         evs[str(i)].append(0.5*( 1 - sum([ (coeff[j]**2)*eigenvalues[int(states[j][-i-1])] for j in range(len(states)) ]))) # Fix this

    #### THIS WORKS ##########

    return evs


