" Basic Molecule Entry "
from abc import ABCMeta, abstractproperty
from typing import List, Union, Protocol

import numpy as np
from monty.json import MSONable
from pymatgen.core.periodic_table import Element


class Entry(MSONable, meta=ABCMeta):
    """
    An abstract entry class to store info for the reaction network
    """

    @abstractproperty
    def energy(self) -> float:
        " The energy of this entry. Doesn't include any entropic effects "

    @abstractproperty
    def formula(self) -> str:
        " The flat string formula for this entry "

    @abstractproperty
    def species(self) -> List[str]:
        " The species in this entry "

    @abstractproperty
    def num_atoms(self) -> int:
        " Number of atoms in the molecule "

    @abstractproperty
    def entry_id(self) -> Union[int, str]:
        " The entry ID "


class HasEntropy(Protocol):
    " Protocol for an Entry-like object that contain Entropy information "

    def entropy(self) -> float:
        " The entropy of this object in eV/K "
