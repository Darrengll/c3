import copy
import numpy as np
from c3.model import Model as Mdl
from c3.c3objs import Quantity as Qty
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


# ##change log
# change 1:change 1q gate envelope from cosine to cosine_norm
# change 2: moving sideband frequency to the top of the file
# change 3:sideband change from 50e6 to 500e6
sideband = 500e6


def create_experiment(lindblad=False, t_final=10e-9, temperature=0.0, drag_alpha=2.36):
    dressed = True
    qubit_lvls = 3
    freq = 5.3e9
    anhar = -212e6
    t1_q1 = 20e-6
    t2star_q1 = 6e-6
    qubit_temp = temperature
    init_temp = temperature
    t_final_id = 1e-9
    sim_res = 100e9
    awg_res = 2e9
    drag_delta = drag_alpha/(2*np.pi*anhar/1e9)
    lo_freq = freq + sideband

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(
            value=freq,
            min_val=5.295e9,
            max_val=5.305e9,
            unit="Hz 2pi",
        ),
        anhar=Qty(
            value=anhar,
            min_val=-380e6,
            max_val=-120e6,
            unit="Hz 2pi",
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(
            value=t1_q1,
            min_val=0.1e-6,
            max_val=100e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star_q1,
            min_val=0.1e-6,
            max_val=100e-6,
            unit='s'
        ),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
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

    init_ground = tasks.InitialiseGround(
        init_temp=Qty(value=init_temp, min_val=0, max_val=0.22, unit="K")
    )
    task_list = [init_ground]
    model = Mdl(phys_components, line_components, task_list)
    model.set_lindbladian(lindblad)
    model.set_dressed(dressed)

    # ### MAKE GENERATOR
    generator = Gnr(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
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

    # ### MAKE GATESET
    gauss_params_single = {
        "amp": Qty(value=0.314*2, min_val=0.05, max_val=0.7, unit="V"),
        "t_final": Qty(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "xy_angle": Qty(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Qty(
            value=-sideband - 3.4e6,
            min_val=-30 * 1e6-sideband,
            max_val=10 * 1e6-sideband,
            unit="Hz 2pi",
        ),
        "delta": Qty(value=drag_delta, min_val=-5, max_val=3, unit=""),
    }

    gauss_env_single = pulse.EnvelopeDrag(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.cosine_norm,
    )
    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            "t_final": Qty(
                value=t_final_id, min_val=0.5e-9, max_val=1.5 * t_final, unit="s"
            )
        },
        shape=envelopes.no_drive,
    )
    carrier_parameters = {
        "freq": Qty(
            value=lo_freq,
            min_val=4.0e9,
            max_val=6e9,
            unit="Hz 2pi",
        ),
        "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    QId = gates.Instruction(
        name="id", t_start=0.0, t_end=t_final_id, channels=["d1"], targets=[0]
    )
    QId.add_component(nodrive_env, "d1")
    QId.add_component(copy.deepcopy(carr), "d1")
    QId.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final_id) % (2 * np.pi)
    )

    rx90p = gates.Instruction(
        name="rx90p", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )
    rx90p.add_component(copy.deepcopy(gauss_env_single), "d1")
    rx90p.add_component(copy.deepcopy(carr), "d1")

    ry90p = gates.Instruction(
        name="ry90p", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )
    ry90p.add_component(copy.deepcopy(gauss_env_single), "d1")
    ry90p.add_component(copy.deepcopy(carr), "d1")

    rx90m = gates.Instruction(
        name="rx90m", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )
    rx90m.add_component(copy.deepcopy(gauss_env_single), "d1")
    rx90m.add_component(copy.deepcopy(carr), "d1")

    ry90m = gates.Instruction(
        name="ry90m", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )
    ry90m.add_component(copy.deepcopy(gauss_env_single), "d1")
    ry90m.add_component(copy.deepcopy(carr), "d1")

    ry90p.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    rx90m.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    ry90m.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    # Instruction.ideal matrices also need to revise

    parameter_map = PMap(
        instructions=[QId, rx90p, ry90p, rx90m, ry90m], model=model, generator=generator
    )

    # ### MAKE EXPERIMENT
    exp = Exp(pmap=parameter_map)
    exp.set_opt_gates(["rx90p[0]", "ry90p[0]", "rx90m[0]", "ry90m[0]"])
    return exp

