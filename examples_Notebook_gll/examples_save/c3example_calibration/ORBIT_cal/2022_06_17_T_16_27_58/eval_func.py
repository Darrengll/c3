def ORBIT_qiskit(params):
    
    populations = []
    results = []
    results_std = []
    shots_nums = []

    # Creating the RB sequences
    seqs = qt_utils.single_length_RB(
            RB_number=RB_number, RB_length=RB_length, target=0
    )
    orbit_exp.set_opt_gates_seq(seqs) # speeds up the simulation of circuits
    circuits = seqs_to_circuit(seqs)
    params_as_dict = [param.asdict() for param in params]


    for circuit in circuits:
        circuit.append(SetParamsGate(params = [params_as_dict, gateset_opt_map]), [0])
        orbit_job = orbit_backend.run(circuit)
        orbit_result = orbit_job.result().data()["state_pops"]
        populations.append(list(orbit_result.values()))
        
    for pop in populations:
        excited_pop = np.array(list(orbit_result.values())[1:]).sum() # total excited states population
        results.append(np.array([excited_pop]))
        results_std.append([0])
        shots_nums.append([shots])

    goal = np.mean(results) # average of the excited state populations from every circuit
    return goal, results, results_std, seqs, shots_nums
