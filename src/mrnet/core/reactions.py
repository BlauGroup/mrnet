import copy
import itertools
import math
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.graphs import MolGraphSplitError

from mrnet.core.extract_reactions import FindConcertedReactions
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.rates import (
    ExpandedBEPRateCalculator,
    ReactionRateCalculator,
    RedoxRateCalculator,
)
from mrnet.utils.constants import KB, PLANCK, ROOM_TEMP
from mrnet.utils.mols import mol_free_energy
from mrnet.utils.reaction import (
    ReactionMappingError,
    generate_atom_mapping_1_1,
    get_reaction_atom_mapping,
)

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith, Mingjian Wen"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


# typing
MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Record_Dict = Dict[str, List[str]]
Atom_Mapping_Dict = Dict[int, int]


# TODO create OneReactantOneProductReaction, subclassing Reaction, but superclassing
#  RedoxReaction and IntramolSingleBondChangeReaction


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
        parameters: Optional[Dict] = None,
        reactants_atom_mapping: Optional[List[Atom_Mapping_Dict]] = None,
        products_atom_mapping: Optional[List[Atom_Mapping_Dict]] = None,
    ):
        self.reactants = reactants
        self.products = products
        self.transition_state = transition_state
        if self.transition_state is None:
            # Provide no reference initially
            self.rate_calculator = None
        else:
            self.rate_calculator = ReactionRateCalculator(
                reactants, products, self.transition_state
            )

        self.reactant_ids = np.array([e.entry_id for e in reactants])
        self.product_ids = np.array([e.entry_id for e in products])

        self.reactant_indices = np.array([r.parameters.get("ind") for r in reactants])
        self.product_indices = np.array([p.parameters.get("ind") for p in products])

        self.parameters = parameters or dict()

        self.reactants_atom_mapping = reactants_atom_mapping
        self.products_atom_mapping = products_atom_mapping

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.reactant_ids or entry.entry_id in self.product_ids

    def update_calculator(
        self,
        transition_state: Optional[MoleculeEntry] = None,
        reference: Optional[Dict] = None,
    ):
        """
        Update the rate calculator with either a transition state (or a new
            transition state) or the thermodynamic properties of a reaction

        Args:
            transition_state (MoleculeEntry): MoleculeEntry referring to a
                transition state
            reference (dict): Dictionary containing relevant thermodynamic
                values for a reference reaction
                Keys:
                    delta_ea: Activation energy
                    delta_ha: Activation enthalpy
                    delta_sa: Activation entropy
                    delta_e: Reaction energy change
                    delta_h: Reaction enthalpy change
                    delta_s: Reaction entropy change
        Returns:
            None
        """

        if transition_state is None:
            if reference is None:
                pass
            else:
                self.rate_calculator = ExpandedBEPRateCalculator(
                    reactants=self.reactants,
                    products=self.products,
                    delta_ea_reference=reference["delta_ea"],
                    delta_ha_reference=reference["delta_ha"],
                    delta_sa_reference=reference["delta_sa"],
                    delta_e_reference=reference["delta_e"],
                    delta_h_reference=reference["delta_h"],
                    delta_s_reference=reference["delta_s"],
                )
        else:
            self.rate_calculator = ReactionRateCalculator(
                self.reactants, self.products, transition_state
            )

    @classmethod
    @abstractmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
    ):
        pass

    @abstractmethod
    def graph_representation(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def set_free_energy(self, temperature=ROOM_TEMP):
        pass

    @abstractmethod
    def set_rate_constant(self):
        pass

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "transition_state": ts,
            "rate_calculator": rc,  # consider writing as_dict/from_dict methods
            "parameters": self.parameters,
            "reactants_atom_mapping": self.reactants_atom_mapping,
            "products_atom_mapping": self.products_atom_mapping,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            if d["rate_calculator"] is None:
                rate_calculator = None
            else:
                rate_calculator = ExpandedBEPRateCalculator.from_dict(
                    d["rate_calculator"]
                )
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        reactants_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["reactants_atom_mapping"]
        ]
        products_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["products_atom_mapping"]
        ]

        reaction = cls(
            reactants,
            products,
            transition_state=ts,
            parameters=d["parameters"],
            reactants_atom_mapping=reactants_atom_mapping,
            products_atom_mapping=products_atom_mapping,
        )
        reaction.rate_calculator = rate_calculator
        return reaction


class RedoxReaction(Reaction):
    """
    A class to define redox reactions as follows:
    One electron oxidation / reduction without change to bonding
        A^n ±e- <-> A^n±1
        Two entries with:
        identical composition
        identical number of edges
        a charge difference of 1
        isomorphic molecule graphs

    Args:
        reactant: MoleculeEntry object
        product: MoleculeEntry object
        inner_reorganization_energy (float): Inner reorganization energy, in eV
        dielectric (float): Dielectric constant of the solvent
        refractive (float): Refractive index of the solvent
        electron_free_energy (float): Free energy of the electron in the electrode, in eV
        radius (float): Solute cavity radius (including inner solvent shell)
        electrode_distance (float): Distance from reactants to electrode, in Angstrom
        parameters (dict): Any additional data about this reaction
        reactant_atom_mapping: atom mapping number dict for reactant
        product_atom_mapping: atom mapping number dict for product
    """

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: MoleculeEntry,
        inner_reorganization_energy=None,
        dielectric=None,
        refractive=None,
        electron_free_energy=None,
        radius=None,
        electrode_distance=None,
        parameters=None,
        reactant_atom_mapping: Optional[Atom_Mapping_Dict] = None,
        product_atom_mapping: Optional[Atom_Mapping_Dict] = None,
    ):
        self.reactant = reactant
        self.product = product
        self.inner_reorganization_energy = inner_reorganization_energy
        self.dielectric = dielectric
        self.refractive = refractive
        self.electron_free_energy = electron_free_energy
        self.radius = radius
        self.electrode_distance = electrode_distance

        rcts_mp = [reactant_atom_mapping] if reactant_atom_mapping is not None else None
        prdts_mp = [product_atom_mapping] if product_atom_mapping is not None else None

        super().__init__(
            [self.reactant],
            [self.product],
            transition_state=None,
            parameters=parameters,
            reactants_atom_mapping=rcts_mp,
            products_atom_mapping=prdts_mp,
        )

        if all(
            [
                x is not None
                for x in [
                    self.inner_reorganization_energy,
                    self.dielectric,
                    self.refractive,
                    self.electron_free_energy,
                    self.radius,
                    self.electrode_distance,
                ]
            ]
        ):
            self.rate_calculator = RedoxRateCalculator(
                [self.reactant],
                [self.product],
                self.inner_reorganization_energy,
                self.dielectric,
                self.refractive,
                self.electron_free_energy,
                self.radius,
                self.electrode_distance,
            )

        # Store necessary mol_entry attributes
        self.reactant_energy = reactant.energy
        self.product_energy = product.energy

        self.reactant_enthalpy = reactant.enthalpy
        self.product_enthalpy = product.enthalpy

        self.reactant_entropy = reactant.entropy
        self.product_entropy = product.entropy

        if product.charge < reactant.charge:
            self.rxn_type_A = "One electron reduction"
            self.rxn_type_B = "One electron oxidation"
        else:
            self.rxn_type_A = "One electron oxidation"
            self.rxn_type_B = "One electron reduction"

        if self.product_energy is not None and self.reactant_energy is not None:
            self.energy_A = self.product_energy - self.reactant_energy
            self.energy_B = self.reactant_energy - self.product_energy
        else:
            self.energy_A = None
            self.energy_B = None

        # These store the free energy at 298.15 K.
        # Initialized to none, generally overwritten by self.set_free_energy()
        self.base_free_energy_A = None
        self.base_free_energy_B = None
        self.k_A = None
        self.k_B = None
        if self.electron_free_energy is not None:
            self.set_free_energy()
            self.set_rate_constant()

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a RedoxReaction class object into graph representation
        (nx.Digraph object). Redox Reaction must be of type 1 reactant -> 1 product

        Returns:
            nx.Digraph object of a single Redox Reaction
        """
        assert len(self.reactant_ids) == len(self.product_ids) == 1
        return general_graph_rep(self)

    def update_calculator(
        self,
        transition_state: Optional[MoleculeEntry] = None,
        reference: Optional[Dict] = None,
    ):
        """
        Update the rate calculator with either a transition state (or a new
            transition state) or the thermodynamic properties of a reaction

        Args:
            transition_state (MoleculeEntry): NOT USED BY THIS METHOD
            reference (dict): Dictionary containing relevant values
                values for a Marcus Theory-based rate calculator
                Keys:
                    lambda_inner: inner solvent reorganization energy, in eV
                    dielectric: dielectric constant of the solvent
                    refractive: refractive index of the solvent
                    electron_free_energy: free energy of the electron, in eV
                    radius: radius of the reactant + inner solvation shell
                    electrode_distance: distance from the reactant to the electrode
        """

        if reference is None:
            pass
        elif self.rate_calculator:
            self.rate_calculator.update_calc(reference)
        else:
            self.rate_calculator = RedoxRateCalculator(
                self.reactants,
                self.products,
                reference["lambda_inner"],
                reference["dielectric"],
                reference["refractive"],
                reference["electron_free_energy"],
                reference["radius"],
                reference["electrode_distance"],
            )

    @classmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
    ) -> List[Reaction]:
        """
        A method to generate all the possible redox reactions from given entries

        Args:
            entries: ReactionNetwork(input_entries).entries,
               entries = {[formula]:{[num_bonds]:{[charge]:MoleculeEntry}}}

        Returns:
            list of RedoxReaction class objects
        """
        reactions = list()  # type: List[Reaction]
        for formula in entries:
            for Nbonds in entries[formula]:
                charges = sorted(entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges) - 1):
                        charge0 = charges[ii]
                        charge1 = charges[ii + 1]
                        if charge1 - charge0 == 1:
                            for entry0 in entries[formula][Nbonds][charge0]:
                                for entry1 in entries[formula][Nbonds][charge1]:
                                    isomorphic, node_mapping = is_isomorphic(
                                        entry0.graph, entry1.graph
                                    )
                                    if isomorphic and node_mapping:
                                        if determine_atom_mappings:
                                            rct_mp, prdt_mp = generate_atom_mapping_1_1(
                                                node_mapping
                                            )
                                            r = cls(
                                                entry0,
                                                entry1,
                                                reactant_atom_mapping=rct_mp,
                                                product_atom_mapping=prdt_mp,
                                            )
                                        else:
                                            r = cls(
                                                entry0,
                                                entry1,
                                            )

                                        reactions.append(r)

        return reactions

    def set_free_energy(self, temperature=ROOM_TEMP):
        """
        A method to determine the free energy of the redox reaction. Note to
        set RedoxReaction.electron_free_energy a value.
        Sets free_energy_A and free_energy_B,
        where free_energy_A is the primary type of the reaction based on the reactant
        and product of the RedoxReaction object, and the backwards of this reaction
        would be free_energy_B.
        Args:
           temperature:

        Returns:
            None
        """
        set_base = False
        if temperature is None or temperature == ROOM_TEMP:
            if (
                self.base_free_energy_A is not None
                and self.base_free_energy_B is not None
            ):
                self.free_energy_A = self.base_free_energy_A
                self.free_energy_B = self.base_free_energy_B
                return
            else:
                set_base = True

        rct_free_energy = mol_free_energy(
            self.reactant_energy,
            self.reactant_enthalpy,
            self.reactant_entropy,
            temp=temperature,
        )
        pro_free_energy = mol_free_energy(
            self.product_energy,
            self.product_enthalpy,
            self.product_entropy,
            temp=temperature,
        )

        if rct_free_energy is not None and pro_free_energy is not None:
            self.free_energy_A = pro_free_energy - rct_free_energy
            self.free_energy_B = rct_free_energy - pro_free_energy

            if self.rxn_type_A == "One electron reduction":
                self.free_energy_A += -self.electron_free_energy
                self.free_energy_B += self.electron_free_energy
            else:
                self.free_energy_A += self.electron_free_energy
                self.free_energy_B += -self.electron_free_energy
        else:
            self.free_energy_A = None
            self.free_energy_B = None

        if set_base:
            self.base_free_energy_A = self.free_energy_A
            self.base_free_energy_B = self.free_energy_B
        return

    def set_rate_constant(self, temperature=ROOM_TEMP):
        if isinstance(self.rate_calculator, RedoxRateCalculator):
            self.k_A = self.rate_calculator.calculate_rate_constant(
                temperature=temperature
            )
            self.k_B = self.rate_calculator.calculate_rate_constant(
                temperature=temperature, reverse=True
            )
        else:
            self.set_free_energy(temperature=temperature)
            if self.electrode_distance is None:
                kappa = 1
            else:
                kappa = np.exp(-1.2 * self.electrode_distance)

            if self.inner_reorganization_energy is None:
                delta_g_a = self.free_energy_A
                delta_g_b = self.free_energy_B
            else:
                lam_reorg = self.inner_reorganization_energy
                delta_g_a = lam_reorg / 4 * (1 + self.free_energy_A / lam_reorg) ** 2
                delta_g_b = lam_reorg / 4 * (1 + self.free_energy_B / lam_reorg) ** 2

            if self.inner_reorganization_energy is None and self.free_energy_A < 0:
                self.k_A = kappa * KB * temperature / PLANCK
            else:
                self.k_A = (
                    kappa
                    * KB
                    * temperature
                    / PLANCK
                    * np.exp(-1 * delta_g_a / (KB * temperature))
                )

            if self.inner_reorganization_energy is None and self.free_energy_B < 0:
                self.k_B = kappa * KB * temperature / PLANCK
            else:
                self.k_B = (
                    kappa
                    * KB
                    * temperature
                    / PLANCK
                    * np.exp(-1 * delta_g_b / (KB * temperature))
                )

    def as_dict(self) -> dict:
        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactant": self.reactant.as_dict(),
            "product": self.product.as_dict(),
            "inner_reorganization_energy": self.inner_reorganization_energy,
            "dielectric": self.dielectric,
            "refractive": self.refractive,
            "electron_free_energy": self.electron_free_energy,
            "radius": self.radius,
            "electrode_distance": self.electrode_distance,
            "rate_calculator": rc,
            "parameters": self.parameters,
            "reactants_atom_mapping": self.reactants_atom_mapping,
            "products_atom_mapping": self.products_atom_mapping,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactant = MoleculeEntry.from_dict(d["reactant"])
        product = MoleculeEntry.from_dict(d["product"])

        if d["rate_calculator"] is None:
            rate_calculator = None
        else:
            rate_calculator = RedoxRateCalculator.from_dict(d["rate_calculator"])

        reactants_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["reactants_atom_mapping"]
        ]
        products_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["products_atom_mapping"]
        ]

        reaction = cls(
            reactant,
            product,
            d["inner_reorganization_energy"],
            d["dielectric"],
            d["refractive"],
            d["electron_free_energy"],
            d["radius"],
            d["electrode_distance"],
            parameters=d["parameters"],
            reactant_atom_mapping=reactants_atom_mapping[0],
            product_atom_mapping=products_atom_mapping[0],
        )
        reaction.rate_calculator = rate_calculator

        return reaction


class IntramolSingleBondChangeReaction(Reaction):
    """
    A class to define intramolecular single bond change as follows:

    Intramolecular formation / breakage of one bond
    A^n <-> B^n
    Two entries with:
        identical composition
        number of edges differ by 1
        identical charge
        removing one of the edges in the graph with more edges yields a graph
        isomorphic to the other entry

    Args:
        reactant: list of single molecular entry
        product: list of single molecular entry
        transition_state: A MoleculeEntry representing a transition state for the
            reaction.
        parameters: Any additional data about this reaction
        reactant_atom_mapping: atom mapping number dict for reactant
        product_atom_mapping: atom mapping number dict for product
    """

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: MoleculeEntry,
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
        reactant_atom_mapping: Optional[Atom_Mapping_Dict] = None,
        product_atom_mapping: Optional[Atom_Mapping_Dict] = None,
    ):
        self.reactant = reactant
        self.product = product

        rcts_mp = [reactant_atom_mapping] if reactant_atom_mapping is not None else None
        prdts_mp = [product_atom_mapping] if product_atom_mapping is not None else None

        super().__init__(
            [self.reactant],
            [self.product],
            transition_state=transition_state,
            parameters=parameters,
            reactants_atom_mapping=rcts_mp,
            products_atom_mapping=prdts_mp,
        )

        # Store necessary mol_entry attributes
        self.reactant_energy = reactant.energy
        self.product_energy = product.energy

        self.reactant_enthalpy = reactant.enthalpy
        self.product_enthalpy = product.enthalpy

        self.reactant_entropy = reactant.entropy
        self.product_entropy = product.entropy

        if product.charge < reactant.charge:
            self.rxn_type_A = "Intramolecular single bond breakage"
            self.rxn_type_B = "Intramolecular single bond formation"
        else:
            self.rxn_type_A = "Intramolecular single bond formation"
            self.rxn_type_B = "Intramolecular single bond breakage"

        if self.product_energy is not None and self.reactant_energy is not None:
            self.energy_A = self.product_energy - self.reactant_energy
            self.energy_B = self.reactant_energy - self.product_energy

        else:
            self.energy_A = None
            self.energy_B = None

        # These store the free energy at 298.15 K.
        # Initialized to none, generally overwritten by self.set_free_energy()
        self.base_free_energy_A = None
        self.base_free_energy_B = None
        self.set_free_energy()
        self.set_rate_constant()

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a IntramolSingleBondChangeReaction class object into
        graph representation (nx.Digraph object).
        IntramolSingleBondChangeReaction must be of type 1 reactant -> 1 product

        Returns:
            nx.Digraph object of a single IntramolSingleBondChangeReaction object
        """
        assert len(self.reactant_ids) == len(self.product_ids) == 1
        return general_graph_rep(self)

    @classmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
    ) -> List[Reaction]:
        reactions = list()  # type: List[Reaction]
        for formula in entries:
            Nbonds_list = sorted(entries[formula].keys())
            if len(Nbonds_list) <= 1:
                continue

            for ii in range(len(Nbonds_list) - 1):
                Nbonds0 = Nbonds_list[ii]
                Nbonds1 = Nbonds_list[ii + 1]
                if Nbonds1 - Nbonds0 != 1:
                    continue

                for charge in entries[formula][Nbonds0]:
                    if charge not in entries[formula][Nbonds1]:
                        continue

                    for entry1 in entries[formula][Nbonds1][charge]:
                        rxns = cls._generate_one(
                            entry1,
                            entries,
                            formula,
                            Nbonds0,
                            charge,
                            determine_atom_mappings,
                            cls,
                        )
                        reactions.extend(rxns)

        return reactions

    @staticmethod
    def _generate_one(
        entry1, entries, formula, Nbonds0, charge, determine_atom_mappings, cls
    ) -> List[Reaction]:
        """
        Helper function to generate reactions for one molecule entry.
        """
        reactions = []
        entry0_set = set()
        for bond in entry1.bonds:
            mg = copy.deepcopy(entry1.mol_graph)
            mg.break_edge(bond[0], bond[1], allow_reverse=True)
            if nx.is_weakly_connected(mg.graph):
                for entry0 in entries[formula][Nbonds0][charge]:
                    isomorphic, node_mapping = is_isomorphic(entry0.graph, mg.graph)
                    if (
                        isomorphic
                        and node_mapping
                        and entry0.entry_id not in entry0_set
                    ):
                        if determine_atom_mappings:
                            rct_mp, prdt_mp = generate_atom_mapping_1_1(node_mapping)
                            r = cls(
                                entry0,
                                entry1,
                                reactant_atom_mapping=rct_mp,
                                product_atom_mapping=prdt_mp,
                            )
                        else:
                            r = cls(
                                entry0,
                                entry1,
                            )

                        reactions.append(r)
                        entry0_set.add(entry0.entry_id)

                        break

        return reactions

    def set_free_energy(self, temperature=ROOM_TEMP):
        """
        A method to determine the free energy of the intramolecular single bond change
        reaction. Sets free_energy_A and free_energy_B
        where free_energy_A is the primary type of the reaction based on
        the reactant and product of the IntramolSingleBondChangeReaction
        object, and the backwards of this reaction would be free_energy_B.

        Args:
            temperature:

        Returns:
            None
        """

        set_base = False
        if temperature is None or temperature == ROOM_TEMP:
            if (
                self.base_free_energy_A is not None
                and self.base_free_energy_B is not None
            ):
                self.free_energy_A = self.base_free_energy_A
                self.free_energy_B = self.base_free_energy_B
                return
            else:
                set_base = True

        rct_free_energy = mol_free_energy(
            self.reactant_energy,
            self.reactant_enthalpy,
            self.reactant_entropy,
            temp=temperature,
        )
        pro_free_energy = mol_free_energy(
            self.product_energy,
            self.product_enthalpy,
            self.product_entropy,
            temp=temperature,
        )

        if rct_free_energy is not None and pro_free_energy is not None:
            self.free_energy_A = pro_free_energy - rct_free_energy
            self.free_energy_B = rct_free_energy - pro_free_energy
        else:
            self.free_energy_A = None
            self.free_energy_B = None

        if set_base:
            self.base_free_energy_A = self.free_energy_A
            self.base_free_energy_B = self.free_energy_B
        return

    def set_rate_constant(self, temperature=ROOM_TEMP):
        if isinstance(self.rate_calculator, ReactionRateCalculator) or isinstance(
            self.rate_calculator, ExpandedBEPRateCalculator
        ):
            self.k_A = self.rate_calculator.calculate_rate_constant(
                temperature=temperature
            )
            self.k_B = self.rate_calculator.calculate_rate_constant(
                temperature=temperature, reverse=True
            )
        else:
            self.set_free_energy(temperature=temperature)

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                self.k_A = KB * temperature / PLANCK
            else:
                self.k_A = (
                    KB * temperature / PLANCK * np.exp(-1 * ga / (KB * temperature))
                )

            if gb < 0:
                self.k_B = KB * temperature / PLANCK
            else:
                self.k_B = (
                    KB * temperature / PLANCK * np.exp(-1 * gb / (KB * temperature))
                )

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "reactant": self.reactant.as_dict(),
            "product": self.product.as_dict(),
            "transition_state": ts,
            "rate_calculator": rc,
            "parameters": self.parameters,
            "reactants_atom_mapping": self.reactants_atom_mapping,
            "products_atom_mapping": self.products_atom_mapping,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactant = MoleculeEntry.from_dict(d["reactant"])
        product = MoleculeEntry.from_dict(d["product"])
        if d["transition_state"] is None:
            ts = None
            if d["rate_calculator"] is None:
                rate_calculator = None
            else:
                rate_calculator = ExpandedBEPRateCalculator.from_dict(
                    d["rate_calculator"]
                )
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        reactants_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["reactants_atom_mapping"]
        ]
        products_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["products_atom_mapping"]
        ]

        reaction = cls(
            reactant,
            product,
            transition_state=ts,
            parameters=d["parameters"],
            reactant_atom_mapping=reactants_atom_mapping[0],
            product_atom_mapping=products_atom_mapping[0],
        )
        reaction.rate_calculator = rate_calculator
        return reaction


# TODO rename to IntermolSingleBondChangeReaction, rename argument `product` to `products`
class IntermolecularReaction(Reaction):
    """
    A class to define intermolecular single bond change as follows:

    Intermolecular breakage / formation of one bond
    A <-> B + C aka B + C <-> A
    Three entries with:
        comp(A) = comp(B) + comp(C)
        charge(A) = charge(B) + charge(C)
        removing one of the edges in A yields two disconnected subgraphs
        that are isomorphic to B and C

    Args:
        reactant: list of single molecular entry
        product: list of two molecular entries
        transition_state: A MoleculeEntry representing a transition state for the reaction.
        parameters: Any additional data about this reaction
        reactant_atom_mapping: atom mapping number dict for reactant
        products_atom_mapping: list of atom mapping number dict for products
    """

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
        reactant_atom_mapping: Optional[Atom_Mapping_Dict] = None,
        products_atom_mapping: Optional[List[Atom_Mapping_Dict]] = None,
    ):
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]

        rcts_mp = [reactant_atom_mapping] if reactant_atom_mapping is not None else None
        prdts_mp = products_atom_mapping if products_atom_mapping is not None else None

        super().__init__(
            [self.reactant],
            [self.product_0, self.product_1],
            transition_state=transition_state,
            parameters=parameters,
            reactants_atom_mapping=rcts_mp,
            products_atom_mapping=prdts_mp,
        )

        # Store necessary mol_entry attributes
        self.reactant_energy = reactant.energy
        self.pro0_energy = product[0].energy
        self.pro1_energy = product[1].energy

        self.reactant_enthalpy = reactant.enthalpy
        self.pro0_enthalpy = product[0].enthalpy
        self.pro1_enthalpy = product[1].enthalpy

        self.reactant_entropy = reactant.entropy
        self.pro0_entropy = product[0].entropy
        self.pro1_entropy = product[1].entropy

        self.rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        self.rxn_type_B = "Molecular formation from one new bond A+B -> C"

        if (
            self.pro1_energy is not None
            and self.pro0_energy is not None
            and self.reactant_energy is not None
        ):
            self.energy_A = self.pro0_energy + self.pro1_energy - self.reactant_energy
            self.energy_B = self.reactant_energy - self.pro0_energy - self.pro1_energy

        else:
            self.energy_A = None
            self.energy_B = None

        # These store the free energy at 298.15 K.
        # Initialized to none, generally overwritten by self.set_free_energy()
        self.base_free_energy_A = None
        self.base_free_energy_B = None
        self.set_free_energy()
        self.set_rate_constant()

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a IntermolecularReaction class object into graph
        representation (nx.Digraph object).
        IntermolecularReaction must be of type 1 reactant -> 2 products

        Returns:
            nx.Digraph object of a single IntermolecularReaction object
        """
        assert len(self.reactant_ids) == 1
        assert len(self.product_ids) == 2
        return general_graph_rep(self)

    @classmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
    ) -> List[Reaction]:
        reactions = list()  # type: List[Reaction]

        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds <= 0:
                    continue

                for charge in entries[formula][Nbonds]:
                    for entry in entries[formula][Nbonds][charge]:
                        rxns = cls._generate_one(
                            entry, entries, charge, determine_atom_mappings, cls
                        )
                        reactions.extend(rxns)

        return reactions

    @staticmethod
    def _generate_one(
        entry, entries, charge, determine_atom_mappings, cls
    ) -> List[Reaction]:
        """
        Helper function to generate reactions for one molecule entry.
        """
        reactions = []
        product_set = set()

        for edge in entry.bonds:
            bond = [(edge[0], edge[1])]
            try:
                frags = entry.mol_graph.split_molecule_subgraphs(
                    bond, allow_reverse=True
                )
                formula0 = frags[0].molecule.composition.alphabetical_formula
                Nbonds0 = len(frags[0].graph.edges())
                formula1 = frags[1].molecule.composition.alphabetical_formula
                Nbonds1 = len(frags[1].graph.edges())

                if (
                    formula0 not in entries
                    or formula1 not in entries
                    or Nbonds0 not in entries[formula0]
                    or Nbonds1 not in entries[formula1]
                ):
                    continue

                for charge0 in entries[formula0][Nbonds0]:
                    charge1 = charge - charge0
                    if charge1 not in entries[formula1][Nbonds1]:
                        continue

                    for entry0 in entries[formula0][Nbonds0][charge0]:
                        isomorphic0, _ = is_isomorphic(frags[0].graph, entry0.graph)
                        if isomorphic0:

                            for entry1 in entries[formula1][Nbonds1][charge1]:
                                isomorphic1, _ = is_isomorphic(
                                    frags[1].graph, entry1.graph
                                )
                                if (
                                    isomorphic1
                                    and frozenset([entry0.entry_id, entry1.entry_id])
                                    not in product_set
                                    and frozenset([entry1.entry_id, entry0.entry_id])
                                    not in product_set
                                ):
                                    if determine_atom_mappings:
                                        (
                                            rcts_mp,
                                            prdts_mp,
                                            num_bond,
                                        ) = get_reaction_atom_mapping(
                                            [entry], [entry0, entry1]
                                        )
                                        if num_bond != 1:
                                            raise ReactionMappingError(
                                                f"Expect 1 bond change; got {num_bond}"
                                            )

                                        r = cls(
                                            entry,
                                            [entry0, entry1],
                                            reactant_atom_mapping=rcts_mp[0],
                                            products_atom_mapping=prdts_mp,
                                        )
                                    else:
                                        r = cls(
                                            entry,
                                            [entry0, entry1],
                                        )

                                    reactions.append(r)
                                    product_set.add(
                                        frozenset([entry0.entry_id, entry1.entry_id])
                                    )
                                    product_set.add(
                                        frozenset([entry1.entry_id, entry0.entry_id])
                                    )

                                    break
                            break
            except MolGraphSplitError:
                pass

        return reactions

    def set_free_energy(self, temperature=ROOM_TEMP):
        """
        A method to determine the free energy of the intermolecular reaction.
        Sets free_energy_A and free_energy_B
        where free_energy_A is the primary type of the reaction based on
        the reactant and product of the IntermolecularReaction
        object, and the backwards of this reaction would be free_energy_B.

        Args:
            temperature:

        Returns:
            None
        """

        set_base = False
        if temperature is None or temperature == ROOM_TEMP:
            if (
                self.base_free_energy_A is not None
                and self.base_free_energy_B is not None
            ):
                self.free_energy_A = self.base_free_energy_A
                self.free_energy_B = self.base_free_energy_B
                return
            else:
                set_base = True

        rct_free_energy = mol_free_energy(
            self.reactant_energy,
            self.reactant_enthalpy,
            self.reactant_entropy,
            temp=temperature,
        )
        pro0_free_energy = mol_free_energy(
            self.pro0_energy, self.pro0_enthalpy, self.pro0_entropy, temp=temperature
        )
        pro1_free_energy = mol_free_energy(
            self.pro1_energy, self.pro1_enthalpy, self.pro1_entropy, temp=temperature
        )

        if (
            rct_free_energy is not None
            and pro0_free_energy is not None
            and pro1_free_energy is not None
        ):
            self.free_energy_A = pro0_free_energy + pro1_free_energy - rct_free_energy
            self.free_energy_B = rct_free_energy - pro0_free_energy - pro1_free_energy
        else:
            self.free_energy_A = None
            self.free_energy_B = None

        if set_base:
            self.base_free_energy_A = self.free_energy_A
            self.base_free_energy_B = self.free_energy_B
        return

    def set_rate_constant(self, temperature=ROOM_TEMP):
        if isinstance(self.rate_calculator, ReactionRateCalculator) or isinstance(
            self.rate_calculator, ExpandedBEPRateCalculator
        ):
            self.k_A = self.rate_calculator.calculate_rate_constant(
                temperature=temperature
            )
            self.k_B = self.rate_calculator.calculate_rate_constant(
                temperature=temperature, reverse=True
            )
        else:
            self.set_free_energy(temperature=temperature)

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                self.k_A = KB * temperature / PLANCK
            else:
                self.k_A = (
                    KB * temperature / PLANCK * np.exp(-1 * ga / (KB * temperature))
                )

            if gb < 0:
                self.k_B = KB * temperature / PLANCK
            else:
                self.k_B = (
                    KB * temperature / PLANCK * np.exp(-1 * gb / (KB * temperature))
                )

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "reactant": self.reactant.as_dict(),
            "product_0": self.product_0.as_dict(),
            "product_1": self.product_1.as_dict(),
            "transition_state": ts,
            "rate_calculator": rc,
            "parameters": self.parameters,
            "reactants_atom_mapping": self.reactants_atom_mapping,
            "products_atom_mapping": self.products_atom_mapping,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactant = MoleculeEntry.from_dict(d["reactant"])
        product_0 = MoleculeEntry.from_dict(d["product_0"])
        product_1 = MoleculeEntry.from_dict(d["product_1"])
        if d["transition_state"] is None:
            ts = None
            if d["rate_calculator"] is None:
                rate_calculator = None
            else:
                rate_calculator = ExpandedBEPRateCalculator.from_dict(
                    d["rate_calculator"]
                )
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        reactants_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["reactants_atom_mapping"]
        ]
        products_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["products_atom_mapping"]
        ]

        reaction = cls(
            reactant,
            [product_0, product_1],
            transition_state=ts,
            parameters=d["parameters"],
            reactant_atom_mapping=reactants_atom_mapping[0],
            products_atom_mapping=products_atom_mapping,
        )
        reaction.rate_calculator = rate_calculator
        return reaction


# TODO rename to CoordinateBondChangeReaction, and rename argument `product` to `products` and
class CoordinationBondChangeReaction(Reaction):
    """
    A class to define coordination bond change as follows:

    Simultaneous formation / breakage of multiple coordination bonds
    A + M <-> AM aka AM <-> A + M
    Three entries with:
        M = Li, Mg, Ca, or Zn
        comp(AM) = comp(A) + comp(M)
        charge(AM) = charge(A) + charge(M)
        removing two M-containing edges in AM yields two disconnected subgraphs that
        are isomorphic to A and M

    Args:
        reactant: molecular entry
        product: list of two molecular entries
        transition_state: a MoleculeEntry representing a transition state
        parameters: any additional data about this reaction
        reactant_atom_mapping: atom mapping number dict for reactant
        products_atom_mapping: list of atom mapping number dict for products
    """

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
        reactant_atom_mapping: Optional[Atom_Mapping_Dict] = None,
        products_atom_mapping: Optional[List[Atom_Mapping_Dict]] = None,
    ):
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]

        rcts_mp = [reactant_atom_mapping] if reactant_atom_mapping is not None else None
        prdts_mp = products_atom_mapping if products_atom_mapping is not None else None

        super().__init__(
            [self.reactant],
            [self.product_0, self.product_1],
            transition_state=transition_state,
            parameters=parameters,
            reactants_atom_mapping=rcts_mp,
            products_atom_mapping=prdts_mp,
        )

        # Store necessary mol_entry attributes
        self.reactant_energy = reactant.energy
        self.pro0_energy = product[0].energy
        self.pro1_energy = product[1].energy

        self.reactant_enthalpy = reactant.enthalpy
        self.pro0_enthalpy = product[0].enthalpy
        self.pro1_enthalpy = product[1].enthalpy

        self.reactant_entropy = reactant.entropy
        self.pro0_entropy = product[0].entropy
        self.pro1_entropy = product[1].entropy

        self.rxn_type_A = "Coordination bond breaking AM -> A+M"
        self.rxn_type_B = "Coordination bond forming A+M -> AM"

        if (
            self.pro1_energy is not None
            and self.pro0_energy is not None
            and self.reactant_energy is not None
        ):
            self.energy_A = self.pro0_energy + self.pro1_energy - self.reactant_energy
            self.energy_B = self.reactant_energy - self.pro0_energy - self.pro1_energy

        else:
            self.energy_A = None
            self.energy_B = None

        # These store the free energy at 298.15 K.
        # Initialized to none, generally overwritten by self.set_free_energy()
        self.base_free_energy_A = None
        self.base_free_energy_B = None
        self.set_free_energy()
        self.set_rate_constant()

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a CoordinationBondChangeReaction class object into graph
        representation (nx.Digraph object).

        CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

        Returns:
             nx.Digraph object of a single CoordinationBondChangeReaction object
        """
        assert len(self.reactant_ids) == 1
        assert len(self.product_ids) == 2
        return general_graph_rep(self)

    @classmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
    ) -> List[Reaction]:

        # find metal entries
        M_entries = dict()  # type: MappingDict
        for formula in entries:
            if formula in ["Li1", "Mg1", "Ca1", "Zn1"]:
                if formula not in M_entries:
                    M_entries[formula] = dict()
                for charge in entries[formula][0]:
                    assert len(entries[formula][0][charge]) == 1
                    M_entries[formula][charge] = entries[formula][0][charge][0]

        reactions = list()  # type: List[Reaction]

        if not M_entries:
            return reactions

        for formula in entries:
            if "Li" in formula or "Mg" in formula or "Ca" in formula or "Zn" in formula:

                for Nbonds in entries[formula]:
                    if Nbonds <= 2:
                        continue

                    for charge in entries[formula][Nbonds]:
                        for entry in entries[formula][Nbonds][charge]:
                            rxns = cls._generate_one(
                                entry, entries, M_entries, determine_atom_mappings, cls
                            )
                            reactions.extend(rxns)

        return reactions

    @staticmethod
    def _generate_one(
        entry, entries, M_entries, determine_atom_mappings, cls
    ) -> List[Reaction]:
        """
        Helper function to generate reactions for one molecule entry.
        """
        reactions = []
        product_set = set()

        nosplit_M_bonds = list()

        for bond in entry.bonds:
            if (
                str(entry.molecule.sites[bond[0]].species) in M_entries
                or str(entry.molecule.sites[bond[1]].species) in M_entries
            ):
                M_bond = (bond[0], bond[1])
                try:
                    entry.mol_graph.split_molecule_subgraphs(
                        [M_bond], allow_reverse=True
                    )
                except MolGraphSplitError:
                    nosplit_M_bonds.append(M_bond)

        bond_pairs = itertools.combinations(nosplit_M_bonds, 2)

        for bond_pair_entry in bond_pairs:
            bond_pair = [
                (int(bond_pair_entry[0][0]), int(bond_pair_entry[0][1])),
                (int(bond_pair_entry[1][0]), int(bond_pair_entry[1][1])),
            ]
            try:
                frags = entry.mol_graph.split_molecule_subgraphs(
                    bond_pair, allow_reverse=True
                )
                M_ind = None
                M_formula = None

                for ii, frag in enumerate(frags):
                    frag_formula = frag.molecule.composition.alphabetical_formula
                    if frag_formula in M_entries:
                        M_ind = ii
                        M_formula = frag_formula
                        break

                if M_ind is None:
                    continue

                for ii, frag in enumerate(frags):
                    if ii == M_ind:
                        continue

                    nonM_formula = frag.molecule.composition.alphabetical_formula
                    nonM_Nbonds = len(frag.graph.edges())
                    if (
                        nonM_formula not in entries
                        or nonM_Nbonds not in entries[nonM_formula]
                    ):
                        continue

                    for nonM_charge in entries[nonM_formula][nonM_Nbonds]:
                        M_charge = entry.charge - nonM_charge
                        if M_charge not in M_entries[M_formula]:
                            continue

                        for nonM_entry in entries[nonM_formula][nonM_Nbonds][
                            nonM_charge
                        ]:
                            isomorphic, _ = is_isomorphic(frag.graph, nonM_entry.graph)
                            if (
                                isomorphic
                                and frozenset(
                                    [
                                        nonM_entry.entry_id,
                                        M_entries[M_formula][M_charge].entry_id,
                                    ]
                                )
                                not in product_set
                            ):
                                this_m = M_entries[M_formula][M_charge]

                                if determine_atom_mappings:
                                    (
                                        rcts_mp,
                                        prdts_mp,
                                        num_bond,
                                    ) = get_reaction_atom_mapping(
                                        [entry], [nonM_entry, this_m]
                                    )

                                    r = cls(
                                        entry,
                                        [nonM_entry, this_m],
                                        reactant_atom_mapping=rcts_mp[0],
                                        products_atom_mapping=prdts_mp,
                                    )
                                else:
                                    r = cls(
                                        entry,
                                        [nonM_entry, this_m],
                                    )
                                reactions.append(r)
                                product_set.add(
                                    frozenset([nonM_entry.entry_id, this_m.entry_id])
                                )

                                break

            except MolGraphSplitError:
                pass

        return reactions

    def set_free_energy(self, temperature=ROOM_TEMP):
        """
        A method to determine the free energy of the coordination bond change reaction
        Sets free_energy_A and free_energy_B
        where free_energy_A is the primary type of the reaction based
        on the reactant and product of the CoordinationBondChangeReaction
        object, and the backwards of this reaction would be free_energy_B.

        Args:
            temperature:
        """

        set_base = False
        if temperature is None or temperature == ROOM_TEMP:
            if (
                self.base_free_energy_A is not None
                and self.base_free_energy_B is not None
            ):
                self.free_energy_A = self.base_free_energy_A
                self.free_energy_B = self.base_free_energy_B
                return
            else:
                set_base = True

        rct_free_energy = mol_free_energy(
            self.reactant_energy,
            self.reactant_enthalpy,
            self.reactant_entropy,
            temp=temperature,
        )
        pro0_free_energy = mol_free_energy(
            self.pro0_energy, self.pro0_enthalpy, self.pro0_entropy, temp=temperature
        )
        pro1_free_energy = mol_free_energy(
            self.pro1_energy, self.pro1_enthalpy, self.pro1_entropy, temp=temperature
        )

        if (
            rct_free_energy is not None
            and pro0_free_energy is not None
            and pro1_free_energy is not None
        ):
            self.free_energy_A = pro0_free_energy + pro1_free_energy - rct_free_energy
            self.free_energy_B = rct_free_energy - pro0_free_energy - pro1_free_energy
        else:
            self.free_energy_A = None
            self.free_energy_B = None

        if set_base:
            self.base_free_energy_A = self.free_energy_A
            self.base_free_energy_B = self.free_energy_B
        return

    def set_rate_constant(self, temperature=ROOM_TEMP):
        if isinstance(self.rate_calculator, ReactionRateCalculator) or isinstance(
            self.rate_calculator, ExpandedBEPRateCalculator
        ):
            self.k_A = self.rate_calculator.calculate_rate_constant(
                temperature=temperature
            )

            self.k_B = self.rate_calculator.calculate_rate_constant(
                temperature=temperature, reverse=True
            )

        else:
            self.set_free_energy(temperature=temperature)

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                self.k_A = KB * temperature / PLANCK
            else:
                self.k_A = (
                    KB * temperature / PLANCK * np.exp(-1 * ga / (KB * temperature))
                )

            if gb < 0:
                self.k_B = KB * temperature / PLANCK
            else:
                self.k_B = (
                    KB * temperature / PLANCK * np.exp(-1 * gb / (KB * temperature))
                )

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "reactant": self.reactant.as_dict(),
            "product_0": self.product_0.as_dict(),
            "product_1": self.product_1.as_dict(),
            "transition_state": ts,
            "rate_calculator": rc,
            "parameters": self.parameters,
            "reactants_atom_mapping": self.reactants_atom_mapping,
            "products_atom_mapping": self.products_atom_mapping,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactant = MoleculeEntry.from_dict(d["reactant"])
        product_0 = MoleculeEntry.from_dict(d["product_0"])
        product_1 = MoleculeEntry.from_dict(d["product_1"])
        if d["transition_state"] is None:
            ts = None
            if d["rate_calculator"] is None:
                rate_calculator = None
            else:
                rate_calculator = ExpandedBEPRateCalculator.from_dict(
                    d["rate_calculator"]
                )
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        reactants_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["reactants_atom_mapping"]
        ]
        products_atom_mapping = [
            {int(k): v for k, v in mp.items()} for mp in d["products_atom_mapping"]
        ]

        reaction = cls(
            reactant,
            [product_0, product_1],
            transition_state=ts,
            parameters=d["parameters"],
            reactant_atom_mapping=reactants_atom_mapping[0],
            products_atom_mapping=products_atom_mapping,
        )
        reaction.rate_calculator = rate_calculator
        return reaction


class ConcertedReaction(Reaction):
    """
    A class to define concerted reactions.
    User can specify either allowing <=1 bond breakage + <=1 bond formation
    OR <=2 bond breakage + <=2 bond formation.
    User can also specify how many electrons are allowed to involve in a
    reaction.
    Can only deal with <= 2 reactants and <=2 products for now.
    For 1 reactant -> 1 product reactions, a maximum 1 bond breakage and 1
    bond formation is allowed,
    even when the user specify "<=2 bond breakage + <=2 bond formation".
    Args:
        reactant([MoleculeEntry]): list of 1-2 molecular entries
        product([MoleculeEntry]): list of 1-2 molecular entries
        transition_state (MoleculeEntry or None): A MoleculeEntry
        representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
    """

    def __init__(
        self,
        reactant: List[MoleculeEntry],
        product: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
        electron_free_energy: Optional[float] = None,
        parameters: Optional[Dict] = None,
        reactants_atom_mapping: Optional[List[Atom_Mapping_Dict]] = None,
        products_atom_mapping: Optional[List[Atom_Mapping_Dict]] = None,
    ):
        """
          Initilizes IntermolecularReaction.reactant to be in the form of a
              MoleculeEntry,
          IntermolecularReaction.product to be in the form of [MoleculeEntry_0,
                                                               MoleculeEntry_1],
          Reaction.reactant to be in the form of a of a list of MoleculeEntry
              of length 1
          Reaction.products to be in the form of a of a list of MoleculeEntry
              of length 2
        Args:
          reactant: MoleculeEntry object
          product: list of MoleculeEntry object of length 2
          transition_state: MoleculeEntry representing the TS for the reaction

        """
        self.reactants = reactant
        self.products = product
        self.electron_free_energy = electron_free_energy
        self.electron_energy = None
        super().__init__(
            reactant,
            product,
            transition_state=transition_state,
            parameters=parameters,
            reactants_atom_mapping=reactants_atom_mapping,
            products_atom_mapping=products_atom_mapping,
        )

        # Store necessary mol_entry attributes
        self.reactant_energy = [r.energy for r in self.reactants]
        self.product_energy = [p.energy for p in self.products]
        self.reactant_enthalpy = [r.enthalpy for r in reactant]
        self.product_enthalpy = [p.enthalpy for p in product]

        self.reactant_entropy = [r.entropy for r in reactant]
        self.product_entropy = [p.entropy for p in product]

        self.reactant_charge = np.sum([r.charge for r in reactant])
        self.product_charge = np.sum([p.charge for p in product])

        self.rxn_type_A = "Concerted"
        self.rxn_type_B = "Concerted"

        if all(nrg is None for nrg in self.reactant_energy) and all(
            nrg is None for nrg in self.product_energy
        ):
            reactant_total_energy = np.sum([nrg for nrg in self.reactant_energy])
            product_total_energy = np.sum([nrg for nrg in self.product_energy])
            self.energy_A = product_total_energy - reactant_total_energy
            self.energy_B = reactant_total_energy - product_total_energy

        else:
            self.energy_A = None
            self.energy_B = None

        # These store the free energy at 298.15 K.
        # Initialized to none, generally overwritten by self.set_free_energy()
        self.base_free_energy_A = None
        self.base_free_energy_B = None
        self.set_free_energy()
        self.set_rate_constant()

    def graph_representation(
        self,
    ) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

        """
        A method to convert a Concerted class object into graph
            representation (nx.Digraph object).
        IntermolecularReaction must be of type 1 reactant -> 2 products
        :return nx.Digraph object of a single IntermolecularReaction object
        """
        assert len(self.reactant_ids) <= 3
        assert len(self.product_ids) <= 3
        if len(self.reactants) == 2 and len(self.products) == 1:
            self.swap_elements()
        g = general_graph_rep(self)
        for node in list(g.nodes):
            if not isinstance(node, int) and g.nodes[node]["free_energy"] > 0:
                g.remove_node(node)
        return g

    def swap_elements(self):
        self.reactants, self.products = self.products, self.reactants
        self.base_free_energy_A, self.base_free_energy_B = (
            self.base_free_energy_B,
            self.base_free_energy_A,
        )
        self.reactant_energy, self.product_energy = (
            self.product_energy,
            self.reactant_energy,
        )
        self.reactant_enthalpy, self.product_enthalpy = (
            self.product_enthalpy,
            self.reactant_enthalpy,
        )
        self.reactant_entropy, self.product_entropy = (
            self.product_entropy,
            self.reactant_entropy,
        )
        self.reactant_charge, self.product_charge = (
            self.product_charge,
            self.reactant_charge,
        )
        self.reactant_ids, self.product_ids = self.product_ids, self.reactant_ids
        self.reactant_indices, self.product_indices = (
            self.product_indices,
            self.reactant_indices,
        )
        self.reactants_atom_mapping, self.products_atom_mapping = (
            self.products_atom_mapping,
            self.reactants_atom_mapping,
        )

    @classmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
        name="nothing",
        read_file=False,
        num_processors=16,
        reaction_type="break2_form2",
        allowed_charge_change=0,
    ) -> List[Reaction]:

        """
        A method to generate all the possible concerted reactions from given
        entries_list.
        Args:
           :param entries(MappingDict)
           :param name(str): The name to put in FindConcertedReactions class. For
                 reading in the files generated from that class.
           :param read_file(bool): whether to read in the file generated from
                 the FindConcertedReactions class.
                 If true, name+'_concerted_rxns.json' has to be present in the
                 running directory. If False, will find concerted reactions
                 on the fly. Note that this will take a couple hours when
                 running on 16 CPU with < 100 entries.
           :param num_processors:
           :param reaction_type: Can choose from "break2_form2" and
                 "break1_form1"
           :param allowed_charge_change: How many charge changes are allowed
                 in a concerted reaction. If zero, sum(reactant total
                 charges) = sun(product total charges). If n(non-zero),
                 allow n-electron redox reactions.
           :return list of IntermolecularReaction class objects
        """
        entries_list = unbucket_mol_entries(entries)
        if read_file:
            all_concerted_reactions = loadfn(name + "_concerted_rxns.json")
        else:
            FCR = FindConcertedReactions(entries_list, name)
            all_concerted_reactions = FCR.get_final_concerted_reactions(
                name, num_processors, reaction_type
            )

        reactions = []
        for reaction in all_concerted_reactions:
            reactants = reaction[0].split("_")
            products = reaction[1].split("_")
            entries0 = [entries_list[int(item)] for item in reactants]
            entries1 = [entries_list[int(item)] for item in products]
            reactant_total_charge = np.sum([item.charge for item in entries0])
            product_total_charge = np.sum([item.charge for item in entries1])
            total_charge_change = product_total_charge - reactant_total_charge
            if abs(total_charge_change) <= allowed_charge_change:
                r = cls(entries0, entries1)
                reactions.append(r)

        return reactions

    def set_free_energy(self, temperature=ROOM_TEMP):
        """
        A method to determine the free energy of the concerted reaction
        Sets free_energy_A and free_energy_B,
        where free_energy_A is the primary type of the reaction based on
        the reactant and product of the ConcertedReaction
        object, and the backwards of this reaction would be free_energy_B.
        Args:
            temperature:
        Returns:
            None
        """

        set_base = False
        if temperature is None or temperature == ROOM_TEMP:
            if (
                self.base_free_energy_A is not None
                and self.base_free_energy_B is not None
            ):
                self.free_energy_A = self.base_free_energy_A
                self.free_energy_B = self.base_free_energy_B
            else:
                set_base = True

        if self.electron_free_energy is None:
            electron_free = 0.0
        else:
            electron_free = self.electron_free_energy

        rct_free_energies = [
            mol_free_energy(
                self.reactant_energy[i],
                self.reactant_enthalpy[i],
                self.reactant_entropy[i],
                temp=temperature,
            )
            for i in range(len(self.reactant_ids))
        ]
        pro_free_energies = [
            mol_free_energy(
                self.product_energy[i],
                self.product_enthalpy[i],
                self.product_entropy[i],
                temp=temperature,
            )
            for i in range(len(self.product_ids))
        ]

        cond_rct = all(el is not None for el in rct_free_energies)
        cond_pro = all(el is not None for el in pro_free_energies)

        if cond_rct and cond_pro:
            reactant_charge = self.reactant_charge
            product_charge = self.product_charge
            reactant_free_energy = np.sum(rct_free_energies)
            product_free_energy = np.sum(pro_free_energies)
            total_charge_change = product_charge - reactant_charge
            self.free_energy_A = (
                product_free_energy
                - reactant_free_energy
                + total_charge_change * electron_free
            )
            self.free_energy_B = (
                reactant_free_energy
                - product_free_energy
                - total_charge_change * electron_free
            )
        else:
            self.free_energy_A = None
            self.free_energy_B = None

        if set_base:
            self.base_free_energy_A = self.free_energy_A
            self.base_free_energy_B = self.free_energy_B
        return

    def set_rate_constant(self, temperature=ROOM_TEMP):
        if isinstance(self.rate_calculator, ReactionRateCalculator) or isinstance(
            self.rate_calculator, ExpandedBEPRateCalculator
        ):
            self.k_A = self.rate_calculator.calculate_rate_constant(
                temperature=temperature
            )
            self.k_B = self.rate_calculator.calculate_rate_constant(
                temperature=temperature, reverse=True
            )
        else:
            self.set_free_energy()

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                self.k_A = KB * temperature / PLANCK
            else:
                self.k_A = (
                    KB * temperature / PLANCK * np.exp(-1 * ga / (KB * temperature))
                )

            if gb < 0:
                self.k_B = KB * temperature / PLANCK
            else:
                self.k_B = (
                    KB * temperature / PLANCK * np.exp(-1 * gb / (KB * temperature))
                )

    def as_dict(self) -> dict:
        if self.transition_state is None:
            ts = None
        else:
            ts = self.transition_state.as_dict()

        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "transition_state": ts,
            "rate_calculator": rc,
            "parameters": self.parameters,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(r) for r in d["reactants"]]
        products = [MoleculeEntry.from_dict(p) for p in d["products"]]
        if d["transition_state"] is None:
            ts = None
            if d["rate_calculator"] is None:
                rate_calculator = None
            else:
                rate_calculator = ExpandedBEPRateCalculator.from_dict(
                    d["rate_calculator"]
                )
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts, parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class MetalHopReaction(Reaction):
    """
    A class to define metal "hopping" bond change as follows:
        Breaking one coordination bond (AM -> A + M) while simultaneously
        forming another (B + M -> BM), with overall stoichiometry
        AM + B <-> BM + A
        Four entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            comp(BM) + comp(B) + comp(M)
            charge(AM) = charge(A) + charge(M)
            charge(BM) = charge(B) + charge(M)
            removing all edges containing M in AM yields two disconnected
            subgraphs that are isomorphic to A and M, and likewise for BM

    NOTE: This class assumes that the reactants and products are in the order:
        reactants: AM, B
        products: BM, A

    Args:
        reactant([MoleculeEntry]): list of single molecular entry
        product([MoleculeEntry]): list of two molecular entries
        transition_state (MoleculeEntry or None): A MoleculeEntry representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
    """

    def __init__(
        self,
        reactants: List[MoleculeEntry],
        products: List[MoleculeEntry],
        metal: MoleculeEntry,
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
        neutral_hop_barrier: Optional[float] = 0.130,
        anion_hop_barrier: Optional[float] = 0.239,
    ):
        """
        Initializes MetalHopReaction.reactant to be in the form of a
            [MoleculeEntry], MetalHopReaction.product to be in the form of
            [MoleculeEntry],

        Args:
            reactants: list of MoleculeEntry objects of length 2
            products: list of MoleculeEntry objects of length 2
            transition_state (MoleculeEntry or None): A MoleculeEntry
                representing a transition state for the reaction.
            parameters (dict): Any additional data about this reaction
            neutral_hop_barrier (float): Energy barrier (in eV) for a metal ion
                to de-coordinate from a neutral species
            anion_hop_barrier (float): Energy barrier (in eV) for a metal ion
                to de-coordinate from an anionic species

        """

        self.reactant_0 = reactants[0]
        self.reactant_1 = reactants[1]

        self.product_0 = products[0]
        self.product_1 = products[1]

        super().__init__(
            [self.reactant_0, self.reactant_1],
            [self.product_0, self.product_1],
            transition_state=transition_state,
            parameters=parameters,
            reactants_atom_mapping=None,
            products_atom_mapping=None,
        )

        self.metal = metal

        self.neutral_hop_barrier = neutral_hop_barrier
        self.anion_hop_barrier = anion_hop_barrier

        # Store necessary mol_entry attributes
        self.rct0_energy = self.reactant_0.energy
        self.rct1_energy = self.reactant_1.energy
        self.pro0_energy = self.product_0.energy
        self.pro1_energy = self.product_1.energy

        self.rct0_enthalpy = self.reactant_0.enthalpy
        self.rct1_enthalpy = self.reactant_1.enthalpy
        self.pro0_enthalpy = self.product_0.enthalpy
        self.pro1_enthalpy = self.product_1.enthalpy

        self.rct0_entropy = self.reactant_0.entropy
        self.rct1_entropy = self.reactant_1.entropy
        self.pro0_entropy = self.product_0.entropy
        self.pro1_entropy = self.product_1.entropy

        self.rxn_type_A = "Metal hopping reaction AM + B <-> A + BM"
        self.rxn_type_B = "Metal hopping reaction AM + B <-> A + BM"

        if (
            self.pro1_energy is not None
            and self.pro0_energy is not None
            and self.rct0_energy is not None
            and self.rct1_energy is not None
        ):
            self.energy_A = (
                self.pro0_energy
                + self.pro1_energy
                - self.rct0_energy
                - self.rct1_energy
            )
            self.energy_B = self.energy_A * -1

        else:
            self.energy_A = None
            self.energy_B = None

        # These store the free energy at 298.15 K.
        # Initialized to none, generally overwritten by self.set_free_energy()
        self.base_free_energy_A = None
        self.base_free_energy_B = None
        self.set_free_energy()
        self.set_rate_constant()

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a CoordinationBondChangeReaction class object
            into graph representation (nx.Digraph object).
        CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

        :return nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        assert len(self.reactant_ids) == 2
        assert len(self.product_ids) == 2
        return general_graph_rep(self)

    @classmethod
    def generate(
        cls,
        entries: MappingDict,
        determine_atom_mappings: bool = False,
    ) -> List[Reaction]:
        reactions = list()  # type: List[Reaction]
        M_entries = dict()  # type: MappingDict
        pairs = list()
        for formula in entries:
            if formula in ["Li1", "Mg1", "Ca1", "Zn1"]:
                if formula not in M_entries:
                    M_entries[formula] = dict()
                for charge in entries[formula][0]:
                    # Only allow cations - neutral/anionic metals probably won't be re-coordinating
                    if charge > 0:
                        assert len(entries[formula][0][charge]) == 1
                        M_entries[formula][charge] = entries[formula][0][charge][0]

        # TODO: implement concept of reaction families for concerted reactions
        if not M_entries:
            return reactions

        for formula in entries:
            if "Li" in formula or "Mg" in formula or "Ca" in formula or "Zn" in formula:
                for Nbonds in entries[formula]:
                    if Nbonds <= 2:
                        continue
                    for charge in entries[formula][Nbonds]:
                        for entry in entries[formula][Nbonds][charge]:
                            pairs.extend(cls._generate_one(entry, entries, M_entries))

        if len(pairs) > 1:
            for combo in itertools.combinations(pairs, 2):
                m_one = combo[0][2]
                m_two = combo[1][2]
                # Only allow if metal ion is the same on both sides
                if m_one.charge == m_two.charge and m_one.formula == m_two.formula:
                    reactions.append(
                        cls(
                            [combo[0][0], combo[1][1]],
                            [combo[1][0], combo[0][1]],
                            m_one,
                        )
                    )

        return reactions

    @staticmethod
    def _generate_one(entry, entries, M_entries):
        pairs = list()
        for aa, atom in enumerate(entry.molecule):
            if str(atom.specie) in ["Li", "Mg", "Zn", "Ca"]:
                edge_list = list()
                for edge in entry.mol_graph.graph.edges():
                    if aa in edge:
                        edge_list.append(edge)

                try:
                    frags = entry.mol_graph.split_molecule_subgraphs(
                        edge_list, allow_reverse=True
                    )
                    M_ind = None
                    M_formula = None
                    for ii, frag in enumerate(frags):
                        frag_formula = frag.molecule.composition.alphabetical_formula
                        if frag_formula in M_entries:
                            M_ind = ii
                            M_formula = frag_formula
                            break
                    if M_ind is not None:
                        for ii, frag in enumerate(frags):
                            if ii != M_ind:
                                nonM_formula = (
                                    frag.molecule.composition.alphabetical_formula
                                )
                                nonM_Nbonds = len(frag.graph.edges())
                                if nonM_formula in entries:
                                    if nonM_Nbonds in entries[nonM_formula]:
                                        for nonM_charge in entries[nonM_formula][
                                            nonM_Nbonds
                                        ]:
                                            M_charge = entry.charge - nonM_charge
                                            if (
                                                M_charge in M_entries[M_formula]
                                                and M_charge > 0
                                            ):
                                                for nonM_entry in entries[nonM_formula][
                                                    nonM_Nbonds
                                                ][nonM_charge]:
                                                    if frag.isomorphic_to(
                                                        nonM_entry.mol_graph
                                                    ):
                                                        pairs.append(
                                                            (
                                                                entry,
                                                                nonM_entry,
                                                                M_entries[M_formula][
                                                                    M_charge
                                                                ],
                                                            )
                                                        )
                                                        break
                except MolGraphSplitError:
                    pass

        return pairs

    def set_free_energy(self, temperature=ROOM_TEMP):
        """
        A method to determine the free energy of the coordination bond change reaction
        Sets free_energy_A and free_energy_B
        where free_energy_A is the primary type of the reaction based
        on the reactant and product of the CoordinationBondChangeReaction
        object, and the backwards of this reaction would be free_energy_B.

        Args:
            temperature:
        """

        set_base = False
        if temperature is None or temperature == ROOM_TEMP:
            if (
                self.base_free_energy_A is not None
                and self.base_free_energy_B is not None
            ):
                self.free_energy_A = self.base_free_energy_A
                self.free_energy_B = self.base_free_energy_B
                return
            else:
                set_base = True

        rct0_free_energy = mol_free_energy(
            self.rct0_energy,
            self.rct0_enthalpy,
            self.rct0_entropy,
            temp=temperature,
        )
        rct1_free_energy = mol_free_energy(
            self.rct1_energy,
            self.rct1_enthalpy,
            self.rct1_entropy,
            temp=temperature,
        )
        pro0_free_energy = mol_free_energy(
            self.pro0_energy,
            self.pro0_enthalpy,
            self.pro0_entropy,
            temp=temperature,
        )
        pro1_free_energy = mol_free_energy(
            self.pro1_energy,
            self.pro1_enthalpy,
            self.pro1_entropy,
            temp=temperature,
        )

        if (
            rct0_free_energy is not None
            and rct1_free_energy is not None
            and pro0_free_energy is not None
            and pro1_free_energy is not None
        ):
            self.free_energy_A = (
                pro0_free_energy
                + pro1_free_energy
                - rct0_free_energy
                - rct1_free_energy
            )
            self.free_energy_B = self.free_energy_A * -1
        else:
            self.free_energy_A = None
            self.free_energy_B = None

        if set_base:
            self.base_free_energy_A = self.free_energy_A
            self.base_free_energy_B = self.free_energy_B
        return

    def set_rate_constant(self, temperature=ROOM_TEMP):

        ga = self.free_energy_A
        gb = self.free_energy_B

        q_no_m_a = self.reactants[0].charge - self.metal.charge
        q_no_m_b = self.products[0].charge - self.metal.charge

        if q_no_m_a == 0:
            barrier_a = self.neutral_hop_barrier
        else:
            barrier_a = self.anion_hop_barrier

        if q_no_m_b == 0:
            barrier_b = self.neutral_hop_barrier
        else:
            barrier_b = self.anion_hop_barrier

        if ga < barrier_a:
            self.k_A = (
                KB * temperature / PLANCK * np.exp(-1 * barrier_a / (KB * temperature))
            )
        else:
            self.k_A = KB * temperature / PLANCK * np.exp(-1 * ga / (KB * temperature))

        if gb < barrier_b:
            self.k_B = (
                KB * temperature / PLANCK * np.exp(-1 * barrier_b / (KB * temperature))
            )
        else:
            self.k_B = KB * temperature / PLANCK * np.exp(-1 * gb / (KB * temperature))

    def as_dict(self) -> dict:

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [r.as_dict() for r in self.reactants],
            "products": [p.as_dict() for p in self.products],
            "reactant_0": self.reactant_0.as_dict(),
            "reactant_1": self.reactant_1.as_dict(),
            "product_0": self.product_0.as_dict(),
            "product_1": self.product_1.as_dict(),
            "metal": self.metal.as_dict(),
            "transition_state": None,
            "rate_calculator": None,
            "parameters": self.parameters,
            "neutral_hop_barrier": self.neutral_hop_barrier,
            "anion_hop_barrier": self.anion_hop_barrier,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(m) for m in d["reactants"]]
        products = [MoleculeEntry.from_dict(m) for m in d["products"]]
        metal = MoleculeEntry.from_dict(d["metal"])

        reaction = cls(
            reactants,
            products,
            metal,
            neutral_hop_barrier=d["neutral_hop_barrier"],
            anion_hop_barrier=d["anion_hop_barrier"],
        )

        return reaction


def general_graph_rep(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into a general graph representation.
    Can handle reactions with arbitrary numbers of reactants and products
    Args:
       :param reaction:(any of the reaction class object, ex. RedoxReaction,
       IntramolSingleBondChangeReaction)
    """
    assert len(reaction.reactant_ids) <= 3
    assert len(reaction.product_ids) <= 3

    # Create the graph object, and define/call appropriate data
    graph = nx.DiGraph()
    rxn_type_A = reaction.rxn_type_A
    rxn_type_B = reaction.rxn_type_B
    energy_A = reaction.energy_A
    energy_B = reaction.energy_B
    reaction.set_free_energy()
    free_energy_A = reaction.free_energy_A
    free_energy_B = reaction.free_energy_B

    # Sort product and reactant indices in ascending order, e.g. A,B or C,D
    pro_sorted_indices = np.argsort(reaction.product_indices)
    rct_sorted_indices = np.argsort(reaction.reactant_indices)

    # Generate the index ordering used to create the node names
    # pro_node_indices = [
    #     [index] + [i for i in pro_sorted_indices if i != index]
    #     for index in range(len(reaction.product_indices))
    # ]
    # rct_node_indices = [
    #     [index] + [i for i in rct_sorted_indices if i != index]
    #     for index in range(len(reaction.reactant_indices))
    # ]

    # Here, create the 'base' names/ids for products and reactants (sorted by index)
    base_pro_name = "+".join(
        [str(reaction.product_indices[i]) for i in pro_sorted_indices]
    )
    base_pro_ids = "+".join([str(reaction.product_ids[i]) for i in pro_sorted_indices])
    base_rct_name = "+".join(
        [str(reaction.reactant_indices[i]) for i in rct_sorted_indices]
    )
    base_rct_ids = "+".join([str(reaction.reactant_ids[i]) for i in rct_sorted_indices])

    fwd_node_name = base_rct_name + "," + base_pro_name
    fwd_node_ids = base_rct_ids + "," + base_pro_ids
    rev_node_name = base_pro_name + "," + base_rct_name
    rev_node_ids = base_pro_ids + "," + base_rct_ids

    # Create fwd reaction node
    if fwd_node_name in graph:
        return
    graph.add_node(
        fwd_node_name,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=fwd_node_ids,
    )

    # Create rev reaction node
    graph.add_node(
        rev_node_name,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=rev_node_ids,
    )
    duplicate_rct = any(
        [
            item
            for item, count in Counter(reaction.reactant_indices).items()
            if count > 1
        ]
    )
    # Create edges w/ reactant molecule nodes
    for reactant in reaction.reactant_indices:
        # Edge from reactant molecule to fwd reaction node
        if not duplicate_rct or not graph.has_edge(int(reactant), fwd_node_name):
            graph.add_edge(
                int(reactant),
                fwd_node_name,
                softplus=softplus(free_energy_A),
                exponent=exponent(free_energy_A),
                rexp=rexp(free_energy_A),
                default_cost=default_cost(free_energy_A),
                weight=1.0,
                PRs=[
                    int(r)
                    for r in reaction.reactant_indices
                    if (r != reactant or (reaction.reactant_indices == r).sum() > 1)
                ],
            )
        # Edge from rev reaction node to reactant molecule
        graph.add_edge(
            rev_node_name,
            int(reactant),
            softplus=0.0,
            exponent=0.0,
            rexp=0.0,
            default_cost=0.0,
            weight=1.0,
        )

    duplicate_prod = any(
        [item for item, count in Counter(reaction.product_indices).items() if count > 1]
    )

    # Create edges w/product molecule nodes
    for product in reaction.product_indices:
        # Edge from product molecule to rev reaction node
        if not duplicate_prod or not graph.has_edge(int(product), rev_node_name):
            graph.add_edge(
                int(product),
                rev_node_name,
                softplus=softplus(free_energy_B),
                exponent=exponent(free_energy_B),
                rexp=rexp(free_energy_B),
                default_cost=default_cost(free_energy_B),
                weight=1.0,
                PRs=[
                    int(p)
                    for p in reaction.product_indices
                    if (p != product or (reaction.product_indices == p).sum() > 1)
                ],
            )
        # Edge from fwd reaction node to product molecule
        graph.add_edge(
            fwd_node_name,
            int(product),
            softplus=0.0,
            exponent=0.0,
            rexp=0.0,
            default_cost=0.0,
            weight=1.0,
        )

    return graph


def softplus(free_energy: float) -> float:
    """
    Method to determine edge weight using softplus cost function
    """
    return float(np.log(1 + (273.0 / 500.0) * np.exp(free_energy)))


def exponent(free_energy: float) -> float:
    """
    Method to determine edge weight using exponent cost function
    """
    return float(np.exp(free_energy))


def rexp(free_energy: float) -> float:
    """
    Method to determine edge weight using exponent(dG/kt) cost function
    """
    if free_energy <= 0:
        d = np.array([[free_energy]], dtype=np.float128)
        r = np.exp(d)
    else:
        d = np.array([[free_energy]], dtype=np.float128)
        r = np.exp(38.94 * d)

    return r[0][0]


def default_cost(free_energy: float) -> float:
    """
    Method to determine edge weight using exponent(dG/kt) + 1 cost function
    """
    return math.exp(min(10.0, free_energy) / (ROOM_TEMP * KB)) + 1


def is_isomorphic(
    g1: nx.MultiDiGraph, g2: nx.MultiDiGraph
) -> Tuple[bool, Union[None, Dict[int, int]]]:
    """
    Check the isomorphic between two graphs g1 and g2 and return the node mapping.

    Args:
        g1: nx graph
        g2: nx graph

    See Also:
        https://networkx.github.io/documentation/stable/reference/algorithms/isomorphism.vf2.html

    Returns:
        is_isomorphic: Whether graphs g1 and g2 are isomorphic.
        node_mapping: Node mapping from g1 to g2 (e.g. {0:2, 1:1, 2:0}), if g1 and g2
            are isomorphic, `None` if not isomorphic.
    """
    nm = iso.categorical_node_match("specie", "ERROR")
    GM = iso.GraphMatcher(g1.to_undirected(), g2.to_undirected(), node_match=nm)
    if GM.is_isomorphic():
        return True, GM.mapping
    else:
        return False, None


# TODO `bucket_mol_entries` and `unbucket_mol_entries` can be moved to mol_entry.py
def bucket_mol_entries(entries: List[MoleculeEntry], keys: Optional[List[str]] = None):
    """
    Bucket molecules into nested dictionaries according to molecule properties
    specified in keys.

    The nested dictionary has keys as given in `keys`, and the innermost value is a
    list. For example, if `keys = ['formula', 'num_bonds', 'charge']`, then the returned
    bucket dictionary is something like:

    bucket[formula][num_bonds][charge] = [mol_entry1, mol_entry2, ...]

    where mol_entry1, mol_entry2, ... have the same formula, number of bonds, and charge.

    Args:
        entries: a list of molecule entries to bucket
        keys: each str should be a molecule property.
            default to ['formula', 'num_bonds', 'charge']

    Returns:
        Nested dictionary of molecule entry bucketed according to keys.
    """
    keys = ["formula", "num_bonds", "charge"] if keys is None else keys

    num_keys = len(keys)
    buckets = {}  # type: ignore
    for m in entries:
        b = buckets
        for i, j in enumerate(keys):
            v = getattr(m, j)
            if i == num_keys - 1:
                b.setdefault(v, []).append(m)
            else:
                b.setdefault(v, {})
            b = b[v]

    return buckets


def unbucket_mol_entries(entries: MappingDict) -> List[MoleculeEntry]:
    """
    Unbucket molecule entries stored in a nested dictionary to a list.

    This is the opposite operation to `bucket_mol_entries()`.

    Args:
        entries: nested dictionaries, e.g.
            bucket[formula][num_bonds][charge] = [mol_entry1, mol_entry2, ...]

    Returns:
        a list of molecule entries
    """

    def unbucket(d):
        for key, v in d.items():
            if isinstance(v, dict):
                unbucket(v)
            elif isinstance(v, Iterable):
                entries_list.extend(v)
            else:
                raise RuntimeError(
                    f"Cannot unbucket molecule entries. Unsupported data type `{type(v)}`"
                )

    entries_list = []  # type: List[MoleculeEntry]
    unbucket(entries)

    return entries_list
