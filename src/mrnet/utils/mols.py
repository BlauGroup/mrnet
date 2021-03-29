# coding: utf-8
# Copyright (c) MR.Net Development Team.

from mrnet.utils.constants import ROOM_TEMP


__author__ = "Aniruddh Khanwale"
__email__ = "akhanwale@lbl.gov"
__version__ = "0.1"
__date__ = "November 2020"


def mol_free_energy(energy: float, enthalpy: float, entropy: float, temp: float = ROOM_TEMP):
    """
    Convert energy/enthalpy/entropy of various units to free energy in eV

    Args:
        energy (float): Electronic energy in Hartree
        enthalpy (float): Enthalpy in kcal/mol
        entropy (float): Entropy in cal/mol-K
        temp (float): Temperature in K. Default is 298.15 (ROOM_TEMP)

    Returns: free energy in eV
    """
    if enthalpy is not None and entropy is not None:
        return energy * 27.21139 + 0.0433641 * enthalpy - temp * entropy * 0.0000433641
    else:
        return None
