from typing import List, Union, Optional
from mrnet.entries import MoleculeEntry
from mrnet.core import HasEntropy, ROOM_TEMPERATURE
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from dataclasses import dataclass
from enum import Enum



@dataclass
class QChemEntry(MoleculeEntry, HasEntropy):
    """
    A molecule entry with properties computed from QChem
    """

    dft_energy: float
    enthalpy: Optional[float] = None
    vibrational_entropy: Optional[float] = None
    solvation_model: str

    @property
    def energy(self) -> float:
        if self.enthalpy is None:
            return self.dft_energy
        return self.dft_energy + self.enthalpy

    @property
    def entropy(self) -> float:
        if self.vibrational_entropy is None:
            return 0
        return self.vibrational_entropy

    @property
    def free_energy(self, temperature: float = ROOM_TEMPERATURE) -> Optional[float]:
    """
    Get the free energy at the given temperature in Kelvin.
    """
        return self.energy - temperature * self.entropy
