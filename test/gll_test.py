# -*- coding: UTF-8 -*-
"""
@Project ：c3 
@File description：
@Author  ：LL Guo
@Date    ：6/13/22 9:17 PM 
"""
from c3.utils.qt_utils import (
    T1_sequence,
    ramsey_sequence,
    ramsey_echo_sequence,
)

length = 4
target = 0
# sequence = T1_sequence(length, target)
# sequence = ramsey_sequence(length, target)
sequence = ramsey_echo_sequence(length, target)
print(sequence)
