# -*- coding: UTF-8 -*-
"""
@Project ：c3 
@File description：
@Author  ：LL Guo
@Date    ：6/13/22 9:49 PM 
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from c3.utils import qt_utils
from c3.libraries.propagation import rk4
from examples_Notebook_gll.examples_save.single_qubit_blackbox_exp import create_experiment
from c3.libraries.fidelities import populations
from c3.libraries.propagation import evaluate_sequences
from Routine_sequences import APE_sequence, Amp_opt_sequence
from opt_map_collection import gateset_opt_map_APE, gateset_opt_map_amp_opt, gateset_opt_map_full
from c3.utils.PlotTool import *
from examples_Notebook_gll.examples_save.single_qubit_blackbox_exp import sideband


#
# def single_qubit_experiment(seq, exp=None, psi_init=None, goal=-1):
#     """
#         Plotting code for time-resolved populations.
#
#         Parameters
#         ----------
#         exp:
#         psi_init: tf.Tensor
#             Initial state or density matrix.
#         seq: list
#             List of operations to apply to the initial state.
#         goal: tf.float64
#             Value of the goal function, if used.
#         """
#     if exp is None:
#         exp = create_experiment()
#     model = exp.pmap.model
#     if psi_init is None:
#         psi_init = model.get_init_state()  # add "init_ground" to the model.tasks
#     # Running seq
#     # exp.pmap.set_parameters(params, opt_map)
#     exp.set_opt_gates_seq(seq)
#     # exp.set_opt_gates("rx90p[0]")
#
#     exp.set_prop_method()
#     # exp.set_prop_method(rk4)
#     exp.compute_propagators()
#     pops = exp.evaluate(seq, psi_init)
#     # # read-out the results
#     # state_labels = {
#     #     "excited": [(1,), (2,)]
#     # }
#     # qubit_label = list(state_labels.keys())[0]
#     # state_labels = state_labels[qubit_label]
#     # state_label = [tuple(l) for l in state_labels]
#     # pops, _ = exp.process(pops, labels=state_label)  # gll:P1+P2
#     # #
#     # #plt.legend(model.state_labels)
#     return pops
#


# # Generate a sequence
# seqs = [['rx90p[0]']]
# # RB_number = 10
# # RB_length = 20
# # seqs = qt_utils.single_length_RB(
# #     RB_number=RB_number, RB_length=RB_length, target=0)
# # seqs = [qt_utils.T1_sequence(length=10000, target=0)]
# pops = single_qubit_experiment(seqs)
# # print(pops[0].numpy())
# # print(np.array(pops))


def single_qubit_experiment(seqs, xlist=None, gateset_opt_map=None, exp=None, psi_init=None):
    """

    Parameters
    ----------
    xlist
    seqs
    gateset_opt_map
    exp
    psi_init

    Returns
    -------

    """

    if exp is None:
        exp = create_experiment()
    exp.set_prop_method()
    # exp.set_prop_method(rk4)
    exp.set_opt_gates_seq(seqs)
    exp.pmap.set_opt_map(gateset_opt_map)

    opt_map = exp.pmap.get_opt_map()
    print('1d para sweep_' + opt_map[0][0])

    model = exp.pmap.model
    if psi_init is None:
        psi_init = model.get_init_state()  # add "init_ground" to the model.tasks
    if (xlist is None) or (gateset_opt_map is None):
        result = exp.evaluate(seqs, psi_init)
    else:
        result = []
        for x in xlist:
            exp.pmap.set_parameters([x])
            # exp.pmap.print_parameters(gateset_opt_map_full)
            propagators = exp.compute_propagators()
            # method 1
            Us = evaluate_sequences(propagators, seqs)
            pop0s = []
            for U in Us:
                pops = populations(tf.matmul(U, psi_init), model.lindbladian)
                pop0s.append(float(pops[0]))  # P0
            # method 2
            pops = exp.evaluate(seqs, psi_init)
            # # read-out the results
            state_labels = {
                "excited": [(1,), (2,)]
            }
            qubit_label = list(state_labels.keys())[0]
            state_labels = state_labels[qubit_label]
            state_label = [tuple(l) for l in state_labels]
            pop1s, _ = exp.process(pops, labels=state_label)  # gll:P1+P2

            result.append(pop0s)  # P0
            # result.append(pop1s)  # P1+P2
    result = np.array(result)
    return result


def Amp_opt(exp, amp_list, theta=np.pi / 2, N=1, target=0):
    """
    Amplitude optimization experiment
    Parameters
    ----------
    exp
    amp_list
    theta: pi or pi/2 pulse
    N:number of pi or pi/2 pulse

    Returns
    -------
    optimized amplitude of pi or pi/2 pulse
    """
    if int(theta / (np.pi / 2)) == 1:
        S = []
        seq = Amp_opt_sequence(length=N, target=target)
        S.append(seq)
    else:
        raise Exception("C3Error:Only pi/2 is allowed now.")
    pops = single_qubit_experiment(xlist=amp_list, seqs=S, gateset_opt_map=gateset_opt_map_amp_opt, exp=exp)
    amp = amp_list[np.argmin(pops[:,0])]
    fig,ax = plotline(amp_list,pops.T, xname='Amp(V)', yname='P0')
    ax.plot([amp]*2, [0, 1], '--')
    return amp


def APE(exp, detune_list, theta=np.pi / 2, Nlist=[6, 7, 8], target=0, plot=True):
    """
    Amplified phase error experiment
    Parameters
    ----------
    exp
    detune_list
    theta
    Nlist
    target
    plot

    Returns
    -------

    """
    if int(theta / (np.pi / 2)) == 1:
        S = []
        for N in Nlist:
            seq = APE_sequence(length=N, target=target)
            S.append(seq)
    else:
        raise Exception("C3Error:Only pi/2 is allowed now.")
    # Transformation from detune_list to freq_offset list, that is to add sideband frequency
    freq_offset_list = detune_list - sideband
    pops = single_qubit_experiment(xlist=freq_offset_list, seqs=S, gateset_opt_map=gateset_opt_map_APE, exp=exp)
    y1 = pops[:, 0]
    y2 = pops[:, 1]
    y3 = pops[:, 2]
    error = abs(y1 - 1) + abs(y2 - 1) + abs(y3 - 1)
    detune = detune_list[np.argmin(error)]
    if plot:
        fig, ax = plotline(detune_list / 1e6, pops.T, xname='Detune(MHz)', yname='P0',
                           labellist=list(map(str, Nlist)))
        ax.plot([detune / 1e6] * 2, [0, 1], '--')
    # update detune
    exp.pmap.set_parameters([detune - sideband], gateset_opt_map_APE)
    return detune - sideband
