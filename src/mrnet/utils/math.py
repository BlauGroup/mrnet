# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


"""
This module provides utilities for basic math operations.
"""

from functools import reduce
import operator


def product(iterable):
    """
    Given some iterator which allows multiplication, perform a product operation on the iterator.
    """

    return reduce(operator.mul, iterable, 1)
