# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


__author__ = "Aniruddh Khanwale"
__email__ = "akhanwale@lbl.gov"
__version__ = "0.1"
__date__ = "November 2020"


def mol_free_energy(energy, enthalpy, entropy, temp: float = 298.15):
    if enthalpy is not None and entropy is not None:
        return energy * 27.21139 + 0.0433641 * enthalpy - temp * entropy * 0.0000433641
    else:
        return None
