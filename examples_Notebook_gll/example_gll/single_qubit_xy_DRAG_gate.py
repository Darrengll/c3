# -*- coding: UTF-8 -*-
"""
@Project ：c3 
@File description：
@Author  ：LL Guo
@Date    ：6/23/22 9:58 AM 
"""
import numpy as np
import tensorflow as tf
from c3.libraries.fidelities import RB, leakage_RB
from examples_Notebook_gll.examples_save.single_qubit_blackbox_exp import create_experiment
from routine_experiments import APE, Amp_opt


def single_qubit_gate_optimize(temperature=0.0, gate_length_1q=10e-9, drag_alpha=2.36,
                               para_sweep=None,
                               optimization_method=None):
    """

    Parameters
    ----------
    temperature
    gate_length_1q
    drag_alpha
    para_sweep
    optimization_method

    Returns
    -------

    """

    lindblad = True
    # create experiment
    exp = create_experiment(lindblad=lindblad, t_final=gate_length_1q,
                            temperature=temperature, drag_alpha=drag_alpha)

    # sweep parameters to determine initial value for optimization
    if para_sweep is None:
        para_sweep = {
            'sweep': False,
            'xname': ['detune', 'amp'],
            'x0': [- 53.002e6, 0.31415],
            'xlist': [np.arange(0.2, 0.5, 0.01), np.arange(0.2, 0.5, 0.01)]
        }
    x0 = para_sweep['x0']
    if para_sweep['sweep'] is True:
        detune_list = para_sweep['xlist'][0]
        amp_list = para_sweep['xlist'][1]
        # APE experiment #
        detune = APE(exp, detune_list, theta=np.pi/2, Nlist=[6, 7, 8])  # Nlist=[1, 4, 7]
        # update detune
        # Rabi experiment(X/2*2) #
        amp = Amp_opt(exp, amp_list, theta=np.pi/2, N=1)
        x0 = [detune, amp]


    # ## optimize single qubit gate
    # set initial value
    exp.pmap.set_opt_map(gateset_opt_map)
    exp.pmap.set_parameters(x0)
    exp.pmap.print_parameters()
    # if optimization_method is None:
    #     # run both 'calibration' and 'optimal control'
    #     optimization_method = 'optimal control'
    #     single_qubit_optimize(exp=exp, optimization_method=optimization_method)
    #
    #     optimization_method = 'calibration'
    #     single_qubit_optimize(exp=exp, optimization_method=optimization_method)
    # elif optimization_method == 'calibration':
    #     single_qubit_optimize(exp=exp, optimization_method=optimization_method)
    # elif optimization_method == 'optimal control':
    #     single_qubit_optimize(exp=exp, optimization_method=optimization_method)
    # else:
    #     print('error')
    #
    # exp.pmap.print_parameters()
    #
    # # Characterize single qubit  gate
    # exp.set_opt_gates(["rx90p[0]", "ry90p[0]", "rx90m[0]", "ry90m[0]"])
    # propagators = exp.compute_propagators()
    #
    # RB(propagators,
    #    min_length=1,
    #    max_length=501,
    #    num_lengths=26,
    #    num_seqs=30,
    #    lindbladian=lindblad,
    #    padding=0
    #    )
    #
    # leakage_RB(
    #     propagators,
    #     min_length=1,
    #     max_length=501,
    #     num_lengths=26,
    #     num_seqs=30,
    #     logspace=False,
    #     lindbladian=lindblad,
    # )
    return 0


from opt_map_collection import gateset_opt_map
import os
import tempfile
from c3.utils import qt_utils
from c3.optimizers.optimalcontrol import OptimalControl
from c3.optimizers.calibration import Calibration
from c3.libraries import algorithms
from c3.libraries.fidelities import unitary_infid_set, lindbladian_unitary_infid_set
from c3.libraries.algorithms import lbfgs


def single_qubit_optimize(exp, optimization_method='calibration'):
    parameter_map = exp.pmap
    if optimization_method == 'calibration':
        def ORBIT_wrapper(p):
            def ORBIT(params, exp, opt_map, qubit_labels, logdir):
                ### ORBIT meta-parameters ###
                RB_length = 100  # How long each sequence is
                RB_number = 30  # How many sequences
                shots = 1000  # How many averages per readout

                ################################
                ### Simulation specific part ###
                ################################

                do_noise = False  # Whether to add artificial noise to the results

                qubit_label = list(qubit_labels.keys())[0]
                state_labels = qubit_labels[qubit_label]
                state_label = [tuple(l) for l in state_labels]

                # Creating the RB sequences #
                seqs = qt_utils.single_length_RB(
                    RB_number=RB_number, RB_length=RB_length, target=0
                )

                # Transmitting the parameters to the experiment #
                exp.pmap.set_parameters(params, opt_map)
                exp.set_opt_gates_seq(seqs)

                # Simulating the gates #
                U_dict = exp.compute_propagators()

                # Running the RB sequences and read-out the results #
                pops = exp.evaluate(seqs)
                pop1s, _ = exp.process(pops, labels=state_label)  # gll:P1+P2

                results = []
                results_std = []
                shots_nums = []

                # Collecting results and statistics, add noise #
                if do_noise:
                    for p1 in pop1s:
                        draws = tf.keras.backend.random_binomial(
                            [shots],
                            p=p1[0],
                            dtype=tf.float64,
                        )
                        results.append([np.mean(draws)])
                        results_std.append([np.std(draws) / np.sqrt(shots)])
                        shots_nums.append([shots])
                else:
                    for p1 in pop1s:
                        results.append(p1.numpy())
                        results_std.append([0])
                        shots_nums.append([shots])

                #######################################
                ### End of Simulation specific part ###
                #######################################

                goal = np.mean(results)
                return goal, results, results_std, seqs, shots_nums

            state_labels = {
                "excited": [(1,), (2,)]
            }
            return ORBIT(
                p, exp, gateset_opt_map, state_labels, "/tmp/c3logs/blackbox"
            )

        alg_options = {
            "popsize": 10,
            "maxfevals": 300,
            "init_point": "True",
            "tolfun": 0.01,
            "spread": 0.25
        }
        # Create a temporary directory to store logfiles, modify as needed
        log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")
        opt = Calibration(
            dir_path=log_dir,
            run_name="ORBIT_cal",
            eval_func=ORBIT_wrapper,
            pmap=parameter_map,
            exp_right=exp,
            algorithm=algorithms.cmaes,
            options=alg_options
        )
        opt.optimize_controls()
        # ## Analysis
        # The following code uses matplotlib to create an ORBIT plot from the logfile.
        import json
        from matplotlib.ticker import MaxNLocator
        from matplotlib import rcParams
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
        rcParams['text.usetex'] = True
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
        if len(goal_function[-1]) < batch_size:
            goal_function.pop(-1)

        fig, ax = plt.subplots(1)
        means = []
        bests = []
        for ii in range(len(goal_function)):
            means.append(np.mean(np.array(goal_function[ii])))
            bests.append(np.min(np.array(goal_function[ii])))
            for pt in goal_function[ii]:
                ax.plot(ii + 1, pt, color='tab:blue', marker="D", markersize=2.5, linewidth=0)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('ORBIT')
        ax.set_xlabel('Iterations')
        ax.plot(
            range(1, len(goal_function) + 1), bests, color="tab:red", marker="D",
            markersize=5.5, linewidth=0, fillstyle='full'
        )
    elif optimization_method == 'optimal control':
        # Create a temporary directory to store logfiles, modify as needed
        log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

        opt = OptimalControl(
            dir_path=log_dir,
            fid_func=lindbladian_unitary_infid_set,
            fid_subspace=["Q1"],
            pmap=parameter_map,
            algorithm=lbfgs,
            options={"maxfun": 300},
            run_name="better_1q_gate"
        )
        opt.set_exp(exp)
        opt.optimize_controls()
        # print(exp.compute_propagators())
        # print(opt.current_best_goal)


if __name__ == '__main__':
    para_sweep = {
        'sweep': True,
        'xname': ['detune', 'amp'],
        'x0': [- 53.002e6, 0.31415],
        'xlist': [np.arange(-30, 1, 1)*1e6, np.arange(0.05, 0.4, 0.01)]
    }
    single_qubit_gate_optimize(temperature=0.0, gate_length_1q=10e-9, drag_alpha=2.36,
                               para_sweep=para_sweep,
                               optimization_method='calibration')   # 'optimal control','calibration'


