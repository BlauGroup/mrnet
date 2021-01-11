import copy
import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.graphs import MolGraphSplitError
from scipy.constants import R, h, k

from mrnet.core.extract_reactions import FindConcertedReactions
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.rates import ExpandedBEPRateCalculator, ReactionRateCalculator, RedoxRateCalculator
from mrnet.utils.graphs import extract_bond_environment
from mrnet.utils.mols import mol_free_energy

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith, Mingjian Wen"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Record_Dict = Dict[str, List[str]]


class Reaction(MSONable, metaclass=ABCMeta):
    """
    Abstract class for subsequent types of reaction class

    Args:
        reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
        products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2.
        transition_state (MoleculeEntry or None): A MoleculeEntry representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
        reactants_atom_mapping: A list of atom mapping number dicts, each dict for one
            reactant with the style {atom_index: atom_mapping_number}, which is the
            same as the rdkit style of atom mapping number. This can be used together
            with `products_atom_mapping` to determine the correspondence of atoms between
            the reactants and the products. Atoms with the same `atom_mapping_number`
            in the reactants and products are the same atom before and after the reaction.
            For example, `reactants_atom_mapping = [{0:1, 1:3}, {0:2, 1:0}]` and
            `products_atom_mapping = [{0:2, 1:1, 2:3}, {0:0}]` means that:
             atom 0 of the first reactant maps to atom 1 of the first product;
             atom 1 of the first reactant maps to atom 2 of the first product;
             atom 0 of the second reactant maps to atom 0 of the first product;
             atom 1 of the second reactant maps to atom 0 of the second product.
        products_atom_mapping: A list of atom mapping number dicts, each dict for one
            product. See `reactants_atom_mapping` for more explanation.
    """

    def __init__(
        self,
        reactants: List[MoleculeEntry],
        products: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
    ):
        self.reactants = reactants
        self.products = products
        self.transition_state = transition_state
       
        self.reactant_ids = np.array([e.entry_id for e in reactants])
        self.product_ids = np.array([e.entry_id for e in products])

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.reactant_ids or entry.entry_id in self.product_ids

    @classmethod
    @abstractmethod
    def generate(cls, **kwargs) -> List[Reaction]:
        pass

    @abstractmethod
    def graph_representation(self) -> nx.DiGraph:
        pass

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "transition_state": ts,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])

        reaction = cls(
            reactants,
            products,
            transition_state=ts,

        )

