from typing import List, Union
from pymatgen.core.composition import Composition
from mrnet.core import Entry
from dataclasses import dataclass
from functools import cached_property


@dataclass
class BasicEntry(Entry):
    """
    A basic entry class
    """

    energy: float
    formula: str
    id: Union[int, str, None] = None

    def entry_id(self) -> Union[int, str]:
        return self.id if self.id is not None else self.formula

    @cached_property
    def species(self) -> List[str]:
        return Composition(self.formula).elements

    @cached_property
    def num_atoms(self) -> int:
        return Composition(self.formula).num_atoms