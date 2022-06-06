#!/usr/bin/env python
# coding: utf-8

# # Complete $C^3$ Loop for a Single Superconducting Qubit

# This notebook demonstrates the key functionalities of the `c3-toolset` package as a 3-step closed loop device bring up process, outlined below:
# 
# - [Model-based Open Loop Optimal Control](#Optimal-Control)
# - [Model-free Closed Loop Hardware Calibration](#Simulated-calibration)
# - [ML-based Characterization & System Identification](#Model-Learning-on-Dataset-from-a-Simulated-Experiment)

# In[1]:


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


# In[2]:


tf.random.set_seed(2441139)
np.random.seed(2441139)


# # Optimal Control
# First, we setup some general parameters for the simulation. We'll simulate a single, three-level Transmon with frequency 5 GHz and anharmonicity -210 MHz. For signal generation, we employ an arbitrary waveform generator with 2 Gigasamples/s (or one pixel per 0.5 ns) which generates an envelope signal that is mixed with a local oscillator shifted 50 Mhz from the qubit resonance. The dynamics simultion will run at 100 Gigasamples/s. The gate time will be 7 ns. 

# In[3]:


dressed = True
qubit_lvls = 3
freq = 5e9
anhar = -210e6
qubit_temp = 0
init_temp = 0
t_final = 7e-9  # Time for single qubit gates
sim_res = 100e9
awg_res = 2e9
sideband = 50e6
lo_freq = 5e9 + sideband


# We create the Qubit and Drive objects, indicating the drive to act on the qubit and collect them in the Model.

# In[4]:


# ### MAKE MODEL
q1 = chip.Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Quantity(
        value=freq,
        min_val=4.995e9,
        max_val=5.005e9,
        unit="Hz 2pi",
    ),
    anhar=Quantity(
        value=anhar,
        min_val=-380e6,
        max_val=-120e6,
        unit="Hz 2pi",
    ),
    hilbert_dim=qubit_lvls,
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
model.set_dressed(dressed)


# Signal processing is emulated by specifying the classical electronics devices and arranging them in a signal chain.

# In[5]:


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

# In[6]:


gauss_params_single = {
    "amp": Quantity(value=0.45, min_val=0.35, max_val=0.5, unit="V"),
    "t_final": Quantity(
        value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
    ),
    "sigma": Quantity(
        value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
    ),
    "xy_angle": Quantity(
        value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
    ),
    "freq_offset": Quantity(
        value=-sideband - 0.5e6,  # IF
        min_val=-60 * 1e6,
        max_val=-40 * 1e6,
        unit="Hz 2pi",
    ),
    "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
}

gauss_env_single = pulse.EnvelopeDrag(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=envelopes.gaussian_nonorm,
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

# In[7]:


gateset_opt_map =   [
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
    ],
    [
      ("rx90p[0]", "d1", "carrier", "framechange"),
      ("ry90p[0]", "d1", "carrier", "framechange"),
      ("rx90m[0]", "d1", "carrier", "framechange"),
      ("ry90m[0]", "d1", "carrier", "framechange")
    ]
  ]

parameter_map.set_opt_map(gateset_opt_map)


# In this example, we optimize 16 parameters in total, where each group of 4 share the same value.

# In[8]:


parameter_map.print_parameters()


# ### Dynamics
# 
# To investigate dynamics, we define the ground state as an initial state.

# In[9]:


import tensorflow as tf


# In[10]:


psi_init = [[0] * 3]
psi_init[0][0] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))


# In[11]:


init_state


# In[12]:


barely_a_seq = ['rx90p[0]']


# In[13]:


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
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
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

# In[14]:


simulation.set_opt_gates("rx90p[0]")
simulation.compute_propagators()


# In[15]:


plot_dynamics(simulation, init_state, barely_a_seq)


# As is, the initial guess for this gate does not provide a high fidelity.

# ### Qiskit circuit with unoptimized gates
# We can also use the simulation as a backend for a qiskit interface and perform the measurement that way.

# In[16]:


# Qiskit related modules
from c3.qiskit import C3Provider
from c3.qiskit.c3_gates import RX90pGate
from qiskit import QuantumCircuit
from qiskit.tools.visualization import plot_histogram


# In[17]:


qc = QuantumCircuit(1)
qc.append(RX90pGate(), [0])
c3_provider = C3Provider()
c3_backend = c3_provider.get_backend("c3_qasm_physics_simulator")
qiskit_exp = copy.deepcopy(simulation)  #
c3_backend.set_c3_experiment(qiskit_exp)
c3_job_unopt = c3_backend.run(qc)
result_unopt = c3_job_unopt.result()
res_pops_unopt = result_unopt.data()["state_pops"]
print("Result from unoptimized gates:") 
pprint(res_pops_unopt)


# In[18]:


plot_histogram(res_pops_unopt, title='Simulation of Qiskit circuit with Unoptimized Gates')


# We select the `OptimalControl` module and set it up to run an `L-BFGS` optimization, using the overlap between ideal and actual unitary as a goal function.

# In[19]:


import os
import tempfile
from c3.optimizers.optimalcontrol import OptimalControl
from c3.libraries.fidelities import unitary_infid_set
from c3.libraries.algorithms import lbfgs

# Create a temporary directory to store logfiles, modify as needed
log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

opt = OptimalControl(
    dir_path=log_dir,
    fid_func=unitary_infid_set,
    fid_subspace=["Q1"],
    pmap=parameter_map,
    algorithm=lbfgs,
    options={"maxfun" : 300},
    run_name="better_X90"
)
opt.set_exp(simulation)


# In[20]:


opt.optimize_controls()


# In[21]:


simulation.compute_propagators()
plot_dynamics(simulation, init_state, barely_a_seq)


# In[22]:


opt.current_best_goal


# In[23]:


parameter_map.print_parameters()


# ### Qiskit circuit with Optimized gates

# In[24]:


qiskit_exp = copy.deepcopy(simulation)
c3_backend.set_c3_experiment(qiskit_exp)
c3_job_unopt = c3_backend.run(qc)
result_unopt = c3_job_unopt.result()
res_pops_unopt = result_unopt.data()["state_pops"]
print("Result from unoptimized gates:") 
pprint(res_pops_unopt)


# In[25]:


plot_histogram(res_pops_unopt, title='Simulation of Qiskit circuit with Optimized Gates')


# # Simulated calibration
# 
# Calibration of control pulses is the process of fine-tuning parameters in a feedback-loop with the experiment. We will simulate this process here by constructing a black-box simulation and interacting with it exactly like an experiment.
# 
# We have manange imports and creation of the black-box the same way as in the previous example in a helper `single_qubit_experiment.py`.

# In[26]:


from single_qubit_experiment import create_experiment

blackbox = create_experiment()


# This blackbox is constructed the same way as above. The difference will be in how we interact with it. First, we decide on what experiment we want to perform and need to specify it as a python function. A general, minimal example would be
# 
# ```
# def exp_communication(params):
#     # Send parameters to experiment controller
#     # and recieve a measurement result.
#     return measurement_result
#  ```
# 
# Again, `params` is a linear vector of bare numbers. The measurement result can be a single number or a set of results. It can also include additional information about statistics, like averaging, standard deviation, etc.

# ### ORBIT - Single-length randomized benchmarking
# The following defines an [ORBIT](https://arxiv.org/abs/1403.0035) procedure. In short, we define sequences of gates that result in an identity gate if our individual gates are perfect. Any deviation from identity gives us a measure of the imperfections in our gates. Our helper `qt_utils` provides these sequences.

# In[27]:


from c3.utils import qt_utils


# In[28]:


qt_utils.single_length_RB(
            RB_number=1, RB_length=5, target=0
    )


# The desired number of 5 gates is selected from a specific set (the Clifford group) and has to be decomposed into the available gate-set. Here, this means 4 gates per Clifford, hence a sequence of 20 gates.

# ### Qiskit Circuits from Sequences

# We create a helper function that translates the sequences we generated in the previous step to Qiskit Circuits which can then be used to communicate with the experiment.

# In[29]:


from c3.qiskit.c3_gates import RX90pGate, RX90mGate, RY90pGate, RY90mGate, SetParamsGate
from qiskit import QuantumCircuit
from typing import List


# In[30]:


QISKIT_GATE_MAP = { "rx90p": RX90pGate, "rx90m": RX90mGate, "ry90p": RY90pGate, "ry90m": RY90mGate}


# In[31]:


def seqs_to_circuit(seqs: List[List[str]]) -> QuantumCircuit:
    circuits = []
    for seq in seqs:
        qc_sec = QuantumCircuit(1, 1)
        for gate in seq:
            qc_sec.append(QISKIT_GATE_MAP[gate[:-3]](), [int(gate[-2])])
        circuits.append(qc_sec)
    return circuits


# In[32]:


seqs = qt_utils.single_length_RB(
            RB_number=3, RB_length=2, target=0
    )
print(seqs)


# In[33]:


circuits = seqs_to_circuit(seqs)
circuits[2].draw()


# ### Communication with the experiment
# Some of the following code is specific to the fact that this a *simulated* calibration. The interface of the `Calibration` to the experiment is simple: parameters in $\rightarrow$ results out. We use the qiskit interface here for submitting jobs to the experiment and getting back the results, which are then processed to calculate the ORBIT goal function.
# 
# The `c3.qiskit` interface provides two pulse level APIs for supplying parameters to the digital twin. One can provide the parameters and the opt_map either in the `options` field of the `backend.run()` call or use the `params` field of the `SetParamsGate` and then append that gate to the circuit. Here, we use the latter one.
# 
# We limit the RB_length and RB_number below to small values to speed up the demonstration. In an actual experiment, these would be much larger values.

# In[34]:


import numpy as np
import tensorflow as tf

### ORBIT meta-parameters ###
RB_length = 12 # How long each sequence is
RB_number = 5  # How many sequences
shots = 1000    # How many averages per readout

orbit_provider = C3Provider()
orbit_exp = blackbox
orbit_backend = orbit_provider.get_backend("c3_qasm_physics_simulator")
orbit_backend.set_c3_experiment(orbit_exp)

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


# ### Optimization
# We first import algorithms and the correct optimizer object.

# In[35]:


from c3.libraries.algorithms import cmaes
from c3.optimizers.calibration import Calibration


# ### Representation of the experiment within $C^3$
# At this point we have to make sure that the gates ("RX90p", etc.) and drive line ("d1") are compatible to the experiment controller operating the blackbox. We mirror the blackbox by creating an experiment in the $C^3$ context:

# It is important to note that in this example, we are transmitting only these four parameters to the experiment. We don't know how the blackbox will implement the pulse shapes and care has to be taken that the parameters are understood on the other end. Optionally, we could specifiy a virtual AWG within $C^3$ and transmit pixilated pulse shapes directly to the physiscal AWG.

# ### Algorithms
# As an optimization algoritm, we choose [CMA-Es](https://en.wikipedia.org/wiki/CMA-ES) and set up some options specific to this algorithm.

# #### Maximum Evaluations
# We set the maximum number of evaluations to 30 for quick demonstration. Ideally around 250 function evaluations are useful for a good optimization. See [docs](https://c3-toolset.readthedocs.io/en/latest/Simulated_calibration.html#analysis) for a longer run.

# In[36]:


alg_options = {
    "popsize" : 10,
    "maxfevals" : 300,
    "init_point" : "True",
    "tolfun" : 0.01,
    "spread" : 0.1
  }


# We define the subspace as both excited states $\{|1>,|2>\}$, assuming read-out can distinguish between 0, 1 and 2.

# In[37]:


state_labels = {
      "excited" : [(1,), (2,)]
  }


# In the real world, this setup needs to be handled in the experiment controller side.
# We construct the optimizer object with the options we setup:

# In[38]:


# Create a temporary directory to store logfiles, modify as needed
log_dir = "c3example_calibration"

opt = Calibration(
    dir_path=log_dir,
    run_name="ORBIT_cal",
    eval_func=ORBIT_qiskit,
    pmap=parameter_map,
    exp_right=simulation,
    algorithm=cmaes,
    options=alg_options
)


# And run the calibration:

# In[39]:


x = parameter_map.get_parameters_scaled()


# In[40]:


opt.optimize_controls()


# ## Analysis
# The following code uses matplotlib to create an ORBIT plot from the logfile.

# In[41]:


import json
from matplotlib.ticker import MaxNLocator
from  matplotlib import rcParams
from matplotlib import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

rcParams['xtick.direction'] = 'in'
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = '--'
rcParams['markers.fillstyle'] = 'none'
rcParams['axes.prop_cycle'] = cycler(
    'linestyle', ["-", "--"]
)

# enable usetex by setting it to True if LaTeX is installed
rcParams['text.usetex'] = False
rcParams['font.size'] = 16
rcParams['font.family'] = 'serif'

logfilename = opt.logdir + "calibration.log"
with open(logfilename, "r") as filename:
    log = filename.readlines()
    

options = json.loads(log[7])

goal_function = []
batch = 0
batch_size = options["popsize"]


eval = 0
for line in log[9:]:
    if line[0] == "{":
        if not eval % batch_size:
            batch = eval // batch_size
            goal_function.append([])
        eval += 1
        point = json.loads(line)
        if 'goal' in point.keys():
            goal_function[batch].append(point['goal'])

# Clean unfinished batch
if len(goal_function[-1])<batch_size:
    goal_function.pop(-1)

fig, ax = plt.subplots(1)
means = []
bests = []
for ii in range(len(goal_function)):
    means.append(np.mean(np.array(goal_function[ii])))
    bests.append(np.min(np.array(goal_function[ii])))
    for pt in goal_function[ii]:
        ax.plot(ii+1, pt, color='tab:blue', marker="D", markersize=2.5, linewidth=0)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel('ORBIT')
ax.set_xlabel('Iterations')
ax.plot(
    range(1, len(goal_function)+1), bests, color="tab:red", marker="D",
    markersize=5.5, linewidth=0, fillstyle='full'
)


# # Model Learning on Dataset from a Simulated Experiment

# In this notebook, we will use a dataset from a simulated experiment, more specifically, the `Simulated_calibration.ipynb` example notebook and perform Model Learning on a simple 1 qubit model.

# ### Imports

# In[42]:


# !pip install -q pandas
import pickle
from pprint import pprint
import copy
import numpy as np
import os
import ast
import hjson
import pandas as pd

from c3.model import Model as Mdl
from c3.c3objs import Quantity as Quantity
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.generator.generator import Generator as Gnr
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.generator.devices as devices
import c3.libraries.hamiltonians as hamiltonians
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes
import c3.libraries.tasks as tasks
from c3.optimizers.modellearning import ModelLearning
from c3.optimizers.sensitivity import Sensitivity


# ## The Dataset

# We first take a look below at the dataset and its properties. To explore more details about how the dataset is generated, please refer to the previous section.

# In[43]:


DATAFILE_PATH = "c3example_calibration/recent/dataset.pickle"


# In[44]:


with open(DATAFILE_PATH, "rb+") as file:
    data = pickle.load(file)


# In[45]:


data.keys()


# Since this dataset was obtained from an ORBIT ([arXiv:1403.0035](https://arxiv.org/abs/1403.0035)) calibration experiment, we have the `opt_map` which will tell us about the gateset parameters being optimized.

# In[46]:


data["opt_map"]


# This `opt_map` implies the calibration experiment focussed on optimizing 
# the amplitude, delta and frequency offset of the gaussian pulse, along 
# with the framechange angle

# Now onto the actual measurement data from the experiment runs

# In[47]:


seqs_data = data["seqs_grouped_by_param_set"]


# **How many experiment runs do we have?**

# In[48]:


len(seqs_data)


# **What does the data from each experiment look like?**
# 
# We take a look at the first data point

# In[49]:


example_data_point = seqs_data[0]


# In[50]:


example_data_point.keys()


# These `keys` are useful in understanding the structure of the dataset. We look at them one by one.

# In[51]:


example_data_point["params"]


# These are the parameters for our parameterised gateset, for the first experiment run. They correspond to the optimization parameters we previously discussed. 

# The `seqs` key stores the sequence of gates that make up this ORBIT calibration experiment. Each ORBIT sequence consists of a set of gates, followed by a measurement operation. This is then repeated for some `n` number of shots (eg, `1000` in this case) and we only store the averaged result along with the standard deviation of these readout shots. Each experiment in turn consists of a number of these ORBIT sequences. The terms *sequence*, *set* and *experiment* are used somewhat loosely here, so we show below what these look like.

# **A single ORBIT sequence**

# In[52]:


example_data_point["seqs"][0][:10]


# **Total number of ORBIT sequences in an experiment**

# In[53]:


len(example_data_point["seqs"])


# **Total number of Measurement results**

# In[54]:


len(example_data_point["results"])


# **The measurement results and the standard deviation look like this**

# In[55]:


example_results = [
    (example_data_point["results"][i], example_data_point["results_std"][i])
    for i in range(len(example_data_point["results"]))
]


# In[56]:


pprint(example_results)


# ## The Model for Model Learning

# An initial model needs to be provided, which we refine by fitting to our calibration data. We do this below. If you want to learn more about what the various components of the model mean, please refer back to the `two_qubits.ipynb` notebook or the documentation.

# In[57]:


exp_opt_map = [[('Q1', 'anhar')], [('Q1', 'freq')]]
parameter_map.set_opt_map(exp_opt_map)


# ## Optimizer 

# In[58]:


datafiles = {"orbit": DATAFILE_PATH} # path to the dataset
run_name = "simple_model_learning" # name of the optimization run
dir_path = "ml_logs" # path to save the learning logs
algorithm = "cma_pre_lbfgs" # algorithm for learning
# this first does a grad-free CMA-ES and then a gradient based LBFGS
options = {
    "cmaes": {
        "popsize": 12,
        "init_point": "True",
        "stop_at_convergence": 10,
        "ftarget": 4,
        "spread": 0.05,
        "stop_at_sigma": 0.01,
    },
    "lbfgs": {"maxfun": 50, "disp": 0},
} # options for the algorithms
sampling = "high_std" # how data points are chosen from the total dataset
batch_sizes = {"orbit": 2} # how many data points are chosen for learning
state_labels = {
    "orbit": [
        [
            1,
        ],
        [
            2,
        ],
    ]
} # the excited states of the qubit model, in this case it is 3-level


# In[59]:


opt = ModelLearning(
    datafiles=datafiles,
    run_name=run_name,
    dir_path=dir_path,
    algorithm=algorithm,
    options=options,
    sampling=sampling,
    batch_sizes=batch_sizes,
    state_labels=state_labels,
    pmap=parameter_map,
)

opt.set_exp(simulation)


# ## Model Learning

# We are now ready to learn from the data and improve our model

# In[60]:


opt.run()


# ### Result of Model Learning

# In[61]:


opt.current_best_goal


# In[62]:


print(opt.pmap.str_parameters(opt.pmap.opt_map))


# ## Visualisation & Analysis of Results

# The Model Learning logs provide a useful way to visualise the learning process and also understand what's going wrong (or right). We now process these logs to read some data points and also plot some visualisations of the Model Learning process

# ### Open, Clean-up and Convert Logfiles

# In[63]:


LOGDIR = opt.logdir


# In[64]:


logfile = os.path.join(LOGDIR, "model_learn.log")
with open(logfile, "r") as f:
    log = f.readlines()


# In[65]:


params_names = [
    item for sublist in (ast.literal_eval(log[3].strip("\n"))) for item in sublist
]
print(params_names)


# In[66]:


data_list_dict = list()
for line in log[9:]:
    if line[0] == "{":
        temp_dict = ast.literal_eval(line.strip("\n"))
        for index, param_name in enumerate(params_names):
            temp_dict[param_name] = temp_dict["params"][index]
        temp_dict.pop("params")
        data_list_dict.append(temp_dict)


# In[67]:


data_df = pd.DataFrame(data_list_dict)


# ### Summary of Logs

# In[68]:


data_df.describe()


# **Best Point**

# In[69]:


best_point_file = os.path.join(LOGDIR, 'best_point_model_learn.log')


# In[70]:


with open(best_point_file, "r") as f:
    best_point_log_dict = hjson.load(f)

best_point_dict = dict(zip(params_names, best_point_log_dict["optim_status"]["params"]))
best_point_dict["goal"] = best_point_log_dict["optim_status"]["goal"]
print(best_point_dict)


# ### Plotting

# We use `matplotlib` to produce the plots below. Please make sure you have the same installed in your python environment.

# In[71]:


from matplotlib.ticker import MaxNLocator
from  matplotlib import rcParams
from matplotlib import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt 


# In[72]:


rcParams["axes.grid"] = True
rcParams["grid.linestyle"] = "--"

# enable usetex by setting it to True if LaTeX is installed
rcParams["text.usetex"] = False
rcParams["font.size"] = 16
rcParams["font.family"] = "serif"


# **In the plots below, the blue line shows the progress of the parameter optimization while the black and the red lines indicate the converged and true value respectively**

# ### Qubit Anharmonicity

# In[73]:


plot_item = "Q1-anhar"
true_value = -210e6

fig, ax = plt.subplots(1)
ax.set_xlabel("Iteration")
ax.set_ylabel(plot_item)
ax.axhline(y=true_value, color="black", linestyle="--")
ax.axhline(y=best_point_dict[plot_item], color="tab:red", linestyle="-.")
ax.plot(data_df[plot_item], color="tab:blue")


# ### Qubit Frequency

# In[74]:


plot_item = "Q1-freq"
true_value = 5e9

fig, ax = plt.subplots(1)
ax.set_xlabel("Iteration")
ax.set_ylabel(plot_item)
ax.axhline(y=true_value, color="black", linestyle="--")
ax.axhline(y=best_point_dict[plot_item], color="tab:red", linestyle="-.")
ax.plot(data_df[plot_item], color="tab:blue")


# ### Goal Function

# In[75]:


plot_item = "goal"

fig, ax = plt.subplots(1)
ax.set_xlabel("Iteration")
ax.axhline(y=best_point_dict[plot_item], color="tab:red", linestyle="-.")
ax.set_ylabel(plot_item)

ax.plot(data_df[plot_item], color="tab:blue")

