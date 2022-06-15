# -*- coding: UTF-8 -*-
"""
@Project ：c3 
@File description：
@Author  ：LL Guo
@Date    ：6/13/22 9:49 PM 
"""
import numpy as np
import matplotlib.pyplot as plt
from c3.utils import qt_utils
from examples_Notebook_gll.examples_save.single_qubit_blackbox_exp import create_experiment


def plot_dynamics(seq, exp=None, psi_init=None, goal=-1):
    """
        Plotting code for time-resolved populations.

        Parameters
        ----------
        exp:
        psi_init: tf.Tensor
            Initial state or density matrix.
        seq: list
            List of operations to apply to the initial state.
        goal: tf.float64
            Value of the goal function, if used.
        """
    if exp is None:
        exp = create_experiment()
    model = exp.pmap.model
    if psi_init is None:
        psi_init = model.get_init_state()  # add "init_ground" to the model.tasks
    # Running seq
    # exp.pmap.set_parameters(params, opt_map)
    exp.set_opt_gates_seq(seq)
    # exp.set_opt_gates("rx90p[0]")

    exp.compute_propagators()
    pops = exp.evaluate(seq, psi_init)
    # # read-out the results
    # state_labels = {
    #     "excited": [(1,), (2,)]
    # }
    # qubit_label = list(state_labels.keys())[0]
    # state_labels = state_labels[qubit_label]
    # state_label = [tuple(l) for l in state_labels]
    # pops, _ = exp.process(pops, labels=state_label)  # gll:P1+P2
    # #
    # #plt.legend(model.state_labels)
    return pops


# Generate a sequence
# seqs = [['rx90p[0]']]
# RB_number = 10
# RB_length = 20
# seqs = qt_utils.single_length_RB(
#     RB_number=RB_number, RB_length=RB_length, target=0)
seqs = qt_utils.T1_sequence(length=100, target=0)
pops = plot_dynamics([seqs])
# print(pops[0].numpy())
print(np.array(pops))
