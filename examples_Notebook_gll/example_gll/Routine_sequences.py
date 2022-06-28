# -*- coding: UTF-8 -*-
"""
@Project ：c3 
@File description：
@Author  ：LL Guo
@Date    ：6/27/22 2:28 PM 
"""
import itertools
import numpy as np
from typing import List
from c3.utils.qt_utils import inverseC, cliffords_decomp


def T1_sequence(length, target):
    """
    Generate a gate sequence to measure relaxation time in a two-qubit chip.

    Parameters
    ----------
    length : int
        Number of Identity gates.
    target : int
        Which qubit is measured.

    Returns
    -------
    list
        Relaxation sequence.

    """
    # wait = ["Id"]
    wait = [f"id[{str(target)}]"]
    prepare_1 = [f"rx90p[{str(target)}]"] * 2
    S = []
    S.extend(prepare_1)
    S.extend(wait * length)
    return S


def ramsey_sequence(length, target):
    """
    Generate a gate sequence to measure dephasing time in a two-qubit chip.

    Parameters
    ----------
    length : int
        Number of Identity gates.
    target : str
        Which qubit is measured. Options: "left" or "right"

    Returns
    -------
    list
        Dephasing sequence.

    """
    # wait = ["id"]
    wait = [f"id[{str(target)}]"]
    rotate_90 = [f"rx90p[{str(target)}]"]
    S = []
    S.extend(rotate_90)
    S.extend(wait * length)
    S.extend(rotate_90)
    return S


def ramsey_echo_sequence(length, target):
    """
    Generate a gate sequence to measure dephasing time in a two-qubit chip including a
    flip in the middle.
    This echo reduce effects detrimental to the dephasing measurement.

    Parameters
    ----------
    length : int
        Number of Identity gates. Should be even.
    target : str
        Which qubit is measured. Options: "left" or "right"

    Returns
    -------
    list
        Dephasing sequence.

    """
    wait = ["id"]
    hlength = length // 2
    rotate_90_p = [f"rx90p[{str(target)}]"]
    rotate_90_m = [f"rx90m[{str(target)}]"]
    S = []
    S.extend(rotate_90_p)
    S.extend(wait * hlength)
    S.extend(rotate_90_p)
    S.extend(rotate_90_p)
    S.extend(wait * hlength)
    S.extend(rotate_90_m)
    return S


def single_length_RB(
    RB_number: int, RB_length: int, target: int = 0
) -> List[List[str]]:
    """Given a length and number of repetitions it compiles Randomized Benchmarking
    sequences.

    Parameters
    ----------
    RB_number : int
        The number of sequences to construct.
    RB_length : int
        The number of Cliffords in each individual sequence.
    target : int
        Index of the target qubit

    Returns
    -------
    list
        List of RB sequences.
    """
    S = []
    for _ in range(RB_number):
        seq = np.random.choice(24, size=RB_length - 1) + 1
        seq = np.append(seq, inverseC(seq))
        seq_gates = []
        for cliff_num in seq:
            g = [f"{c}[{target}]" for c in cliffords_decomp[cliff_num - 1]]
            seq_gates.extend(g)
        S.append(seq_gates)
    return S


def APE_sequence(length, target):
    """
    Generate a gate sequence to implement APE(amplified phase error) experiment.

    Parameters
    ----------
    length : int
        Number of Identity gates consists of [X/2, -X/2].
    target : int
        Which qubit is measured.

    Returns
    -------
    list
        APE sequence.

    """
    prepare_1 = [f"rx90p[{str(target)}]", f"rx90m[{str(target)}]"]
    S = []
    S.extend(prepare_1 * length)
    return S


def Amp_opt_sequence(length, target):
    """
    Generate a gate sequence to implement amplitude optimization experiment,
    corresponding to rabi experiment when length=1

    Parameters
    ----------
    length : int
        Number of X pulse consists of [X/2, X/2].
    target : int
        Which qubit is measured.

    Returns
    -------
    list
        APE sequence.

    """
    prepare_1 = [f"rx90p[{str(target)}]", f"rx90p[{str(target)}]"]
    S = []
    S.extend(prepare_1 * length)
    return S


if __name__ == "__main__":
    # print(APE_sequence(length=5, target=0))
    print(Amp_opt_sequence(length=1, target=0))
