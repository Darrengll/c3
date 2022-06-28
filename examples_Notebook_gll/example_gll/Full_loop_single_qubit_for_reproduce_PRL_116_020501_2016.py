#!/usr/bin/env python
# coding: utf-8

# # Complete $C^3$ Loop for a Single Superconducting Qubit

# This notebook demonstrates the key functionalities of the `c3-toolset` package as a 3-step closed loop device bring up process, outlined below:
# 
# - [Model-based Open Loop Optimal Control](#Optimal-Control)
# - [Model-free Closed Loop Hardware Calibration](#Simulated-calibration)
# - [ML-based Characterization & System Identification](#Model-Learning-on-Dataset-from-a-Simulated-Experiment)


import copy
from pprint import pprint
import numpy as np
import tensorflow as tf
from c3.model import Model
from c3.c3objs import Quantity
from c3.parametermap import ParameterMap
from c3.experiment import Experiment
from c3.generator.generator import Generator
from c3.signal import gates, pulse
from c3.generator import devices
from c3.libraries import chip, hamiltonians, envelopes

tf.random.set_seed(2441139)
np.random.seed(2441139)


# # Optimal Control
# First, we setup some general parameters for the simulation. We'll simulate a single, three-level Transmon with frequency 5 GHz and anharmonicity -210 MHz. For signal generation, we employ an arbitrary waveform generator with 2 Gigasamples/s (or one pixel per 0.5 ns) which generates an envelope signal that is mixed with a local oscillator shifted 50 Mhz from the qubit resonance. The dynamics simultion will run at 100 Gigasamples/s. The gate time will be 7 ns. 

# In[5]:

def full_run():
    lindblad = True
    dressed = True
    qubit_lvls = 3
    freq = 5.3e9
    anhar = -212e6
    t1_q1 = 20e-6
    t2star_q1 = 6e-6
    qubit_temp = 0
    init_temp = 0
    t_final = 10e-9  # Time for single qubit gates
    sim_res = 100e9
    awg_res = 2e9
    sideband = 50e6
    lo_freq = freq + sideband

    # We create the Qubit and Drive objects, indicating the drive to act on the qubit and collect them in the Model.

    # In[7]:

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Quantity(
            value=freq,
            min_val=5.295e9,
            max_val=5.305e9,
            unit="Hz 2pi",
        ),
        anhar=Quantity(
            value=anhar,
            min_val=-380e6,
            max_val=-120e6,
            unit="Hz 2pi",
        ),
        hilbert_dim=qubit_lvls,
        t1=Quantity(
            value=t1_q1,
            min_val=0.1e-6,
            max_val=90e-6,
            unit='s'
        ),
        t2star=Quantity(
            value=t2star_q1,
            min_val=0.1e-6,
            max_val=90e-3,
            unit='s'
        ),
        temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    phys_components = [q1]
    line_components = [drive]

    model = Model(phys_components, line_components)
    model.set_lindbladian(lindblad)
    model.set_dressed(dressed)

    # Signal processing is emulated by specifying the classical electronics devices and arranging them in a signal chain.

    # In[8]:

    generator = Generator(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains={
            "d1": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"],
            }
        },
    )

    # Lastly, we specify the pulse parametrization and construct the gate-set, consisting of four rotations around the x and y axis of the Bloch sphere by 90 degrees in positive and negative directions.

    # In[9]:

    gauss_params_single = {
        "amp": Quantity(value=0.314, min_val=0.2, max_val=0.5, unit="V"),   # 0.314
        "t_final": Quantity(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "xy_angle": Quantity(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=-sideband - 3.417e6,   # - 3.417e6
            min_val=-60 * 1e6,
            max_val=-40 * 1e6,
            unit="Hz 2pi",
        ),
        "delta": Quantity(value=-1.771, min_val=-5, max_val=3, unit=""),   # -1.771
    }

    gauss_env_single = pulse.EnvelopeDrag(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.cosine,
    )
    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            "t_final": Quantity(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            )
        },
        shape=envelopes.no_drive,
    )
    carrier_parameters = {
        "freq": Quantity(
            value=lo_freq,
            min_val=4.5e9,
            max_val=6e9,
            unit="Hz 2pi",
        ),
        "framechange": Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    rx90p = gates.Instruction(
        name="rx90p", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )
    QId = gates.Instruction(
        name="id", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )

    rx90p.add_component(gauss_env_single, "d1")
    rx90p.add_component(carr, "d1")
    QId.add_component(nodrive_env, "d1")
    QId.add_component(copy.deepcopy(carr), "d1")
    QId.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) % (2 * np.pi)
    )
    ry90p = copy.deepcopy(rx90p)
    ry90p.name = "ry90p"
    rx90m = copy.deepcopy(rx90p)
    rx90m.name = "rx90m"
    ry90m = copy.deepcopy(rx90p)
    ry90m.name = "ry90m"
    ry90p.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    rx90m.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    ry90m.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)

    parameter_map = ParameterMap(
        instructions=[QId, rx90p, ry90p, rx90m, ry90m], model=model, generator=generator
    )

    # ### MAKE EXPERIMENT
    simulation = Experiment(pmap=parameter_map)

    # To specify the optimization task, we collect the pulse parameters we wish to optimize in the `opt_map`, a nested list that groups parameters that share the same value.

    # In[10]:

    gateset_opt_map = [
        [
            ("rx90p[0]", "d1", "gauss", "amp"),
            ("ry90p[0]", "d1", "gauss", "amp"),
            ("rx90m[0]", "d1", "gauss", "amp"),
            ("ry90m[0]", "d1", "gauss", "amp")
        ],
        [
            ("rx90p[0]", "d1", "gauss", "delta"),
            ("ry90p[0]", "d1", "gauss", "delta"),
            ("rx90m[0]", "d1", "gauss", "delta"),
            ("ry90m[0]", "d1", "gauss", "delta")
        ],
        [
            ("rx90p[0]", "d1", "gauss", "freq_offset"),
            ("ry90p[0]", "d1", "gauss", "freq_offset"),
            ("rx90m[0]", "d1", "gauss", "freq_offset"),
            ("ry90m[0]", "d1", "gauss", "freq_offset")
        ]
    ]

    parameter_map.set_opt_map(gateset_opt_map)

    # In this example, we optimize 16 parameters in total, where each group of 4 share the same value.

    # In[29]:

    parameter_map.print_parameters()

    # ### Dynamics
    #
    # To investigate dynamics, we define the ground state as an initial state.

    # In[30]:

    import tensorflow as tf

    # In[31]:

    psi_init = [[0] * 3]
    psi_init[0][0] = 1
    init_state = tf.transpose(tf.constant(psi_init, tf.complex128))

    # In[32]:

    init_state

    # In[33]:

    barely_a_seq = ['rx90p[0]']

    # In[34]:

    # !pip install -q matplotlib

    import matplotlib.pyplot as plt

    def plot_dynamics(exp, psi_init, seq, goal=-1):
        """
            Plotting code for time-resolved populations.

            Parameters
            ----------
            psi_init: tf.Tensor
                Initial state or density matrix.
            seq: list
                List of operations to apply to the initial state.
            goal: tf.float64
                Value of the goal function, if used.
            debug: boolean
                If true, return a matplotlib figure instead of saving.
            """
        model = exp.pmap.model
        dUs = exp.partial_propagators
        psi_t = psi_init.numpy()
        pop_t = exp.populations(psi_t, model.lindbladian)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = exp.populations(psi_t, model.lindbladian)
                pop_t = np.append(pop_t, pops, axis=1)

        fig, axs = plt.subplots(1, 1)
        ts = exp.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid(linestyle="--")
        axs.tick_params(
            direction="in", left=True, right=True, top=True, bottom=True
        )
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(model.state_labels)
        pass

    # We set the simulation up to only run the `rx90p[0]` gate and simulate the dynamics.

    # In[35]:

    simulation.set_opt_gates("rx90p[0]")
    simulation.compute_propagators()

    # In[36]:

    # plot_dynamics(simulation, init_state, barely_a_seq)

    # As is, the initial guess for this gate does not provide a high fidelity.

    # ### Qiskit circuit with unoptimized gates
    # We can also use the simulation as a backend for a qiskit interface and perform the measurement that way.

    # In[37]:

    # Qiskit related modules
    from c3.qiskit import C3Provider
    from c3.qiskit.c3_gates import RX90pGate
    from qiskit import QuantumCircuit
    from qiskit.tools.visualization import plot_histogram

    # # In[38]:
    #
    #
    # qc = QuantumCircuit(1)
    # qc.append(RX90pGate(), [0])
    # c3_provider = C3Provider()
    # c3_backend = c3_provider.get_backend("c3_qasm_physics_simulator")
    # qiskit_exp = copy.deepcopy(simulation)
    # c3_backend.set_c3_experiment(qiskit_exp)
    # c3_job_unopt = c3_backend.run(qc)
    # result_unopt = c3_job_unopt.result()
    # res_pops_unopt = result_unopt.data()["state_pops"]
    # print("Result from unoptimized gates:")
    # pprint(res_pops_unopt)
    #
    #
    # # In[39]:
    #
    #
    # plot_histogram(res_pops_unopt, title='Simulation of Qiskit circuit with Unoptimized Gates')

    # We select the `OptimalControl` module and set it up to run an `L-BFGS` optimization, using the overlap between ideal and actual unitary as a goal function.

    # In[40]:

    import os
    import tempfile
    from c3.optimizers.optimalcontrol import OptimalControl
    from c3.libraries.fidelities import unitary_infid_set, lindbladian_unitary_infid_set
    from c3.libraries.algorithms import lbfgs

    # Create a temporary directory to store logfiles, modify as needed
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=lindbladian_unitary_infid_set,
        fid_subspace=["Q1"],
        pmap=parameter_map,
        algorithm=lbfgs,
        options={"maxfun": 300},
        run_name="better_X90"
    )
    opt.set_exp(simulation)

    # In[41]:

    opt.optimize_controls()

    # In[42]:

    simulation.compute_propagators()
    # plot_dynamics(simulation, init_state, barely_a_seq)

    # In[43]:

    opt.current_best_goal

    # In[44]:

    parameter_map.print_parameters()

    # ### Qiskit circuit with Optimized gates

    # In[45]:

    # qiskit_exp = copy.deepcopy(simulation)
    # c3_backend.set_c3_experiment(qiskit_exp)
    # c3_job_unopt = c3_backend.run(qc)
    # result_unopt = c3_job_unopt.result()
    # res_pops_unopt = result_unopt.data()["state_pops"]
    # print("Result from unoptimized gates:")
    # pprint(res_pops_unopt)
    #
    #
    # # In[46]:
    #
    #
    # plot_histogram(res_pops_unopt, title='Simulation of Qiskit circuit with Optimized Gates')

    # # RB test_llguo

    # In[85]:

    from c3.utils import qt_utils

    # In[86]:

    seqs = qt_utils.single_length_RB(RB_number=1, RB_length=20, target=0)
    print(seqs)

    # In[87]:

    simulation.set_opt_gates_seq(seqs)

    # In[88]:

    simulation.opt_gates

    # In[89]:

    propagators = simulation.compute_propagators()
    # propagators

    # In[90]:

    # print(tf.matmul(propagators['ry90p[0]'],propagators['ry90m[0]']))
    # print(tf.matmul(propagators['rx90p[0]'],propagators['rx90m[0]']))
    # print(tf.matmul(propagators['ry90p[0]'],propagators['ry90m[0]'])-tf.matmul(propagators['rx90p[0]'],propagators['rx90m[0]']))

    # In[105]:

    from c3.libraries.fidelities import RB, leakage_RB

    # In[108]:

    RB(propagators,
       min_length=1,
       max_length=501,
       num_lengths=26,
       num_seqs=30,
       lindbladian=lindblad,
       padding=0
       )

    # In[109]:

    leakage_RB(
        propagators,
        min_length=1,
        max_length=501,
        num_lengths=26,
        num_seqs=30,
        logspace=False,
        lindbladian=lindblad,
    )


if __name__ == "__main__":
    full_run()
