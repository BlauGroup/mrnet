from abc import ABCMeta, abstractmethod
import copy
import itertools
import numpy as np
from scipy.constants import h, k, R
from collections.abc import Iterable
from typing import Dict, Tuple, Optional, Union, List

import networkx as nx
import networkx.algorithms.isomorphism as iso

from monty.json import MSONable
from monty.serialization import loadfn

from pymatgen.analysis.graphs import MolGraphSplitError
from mrnet.core.mol_entry import MoleculeEntry

from mrnet.core.rates import (
    ReactionRateCalculator,
    ExpandedBEPRateCalculator,
    RedoxRateCalculator,
)


__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith, Mingjian Wen"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Energy_Dict = Dict[str, float]
Mapping_ReactionType_Dict = Dict[str, str]
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
        reactants_atom_mapping: List[Atom_Mapping_Dict] = None,
        products_atom_mapping: List[Atom_Mapping_Dict] = None,
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

        self.reactant_ids = [e.entry_id for e in self.reactants]
        self.product_ids = [e.entry_id for e in self.products]
        self.entry_ids = {e.entry_id for e in self.reactants + self.products}

        self.parameters = parameters or dict()

        self.reactants_atom_mapping = reactants_atom_mapping
        self.products_atom_mapping = products_atom_mapping

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.entry_ids

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
    def generate(cls, entries: MappingDict):
        pass

    @abstractmethod
    def graph_representation(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def reaction_type(self) -> Mapping_ReactionType_Dict:
        pass

    @abstractmethod
    def energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def rate_constant(self) -> Mapping_Energy_Dict:
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
            "rate_calculator": rc,
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


Mapping_Family_Dict = Dict[str, Dict[int, Dict[int, List[Reaction]]]]


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
        electrode_dist (float): Distance from reactants to electrode, in Angstrom
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
        electrode_dist=None,
        parameters=None,
        reactant_atom_mapping: Atom_Mapping_Dict = None,
        product_atom_mapping: Atom_Mapping_Dict = None,
    ):
        self.reactant = reactant
        self.product = product

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

        self.inner_reorganization_energy = inner_reorganization_energy
        self.dielectric = dielectric
        self.refractive = refractive
        self.electron_free_energy = electron_free_energy
        self.radius = radius
        self.electrode_dist = electrode_dist

        if all(
            [
                x is not None
                for x in [
                    self.inner_reorganization_energy,
                    self.dielectric,
                    self.refractive,
                    self.electron_free_energy,
                    self.radius,
                    self.electrode_dist,
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
                self.electrode_dist,
            )

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a RedoxReaction class object into graph representation
        (nx.Digraph object). Redox Reaction must be of type 1 reactant -> 1 product

        Returns:
            nx.Digraph object of a single Redox Reaction
        """

        return graph_rep_1_1(self)

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
                    electrode_dist: distance from the reactant to the electrode
        """

        if reference is None:
            pass
        else:
            self.rate_calculator = RedoxRateCalculator(
                [self.reactant],
                [self.product],
                reference["lambda_inner"],
                reference["dielectric"],
                reference["refractive"],
                reference["electron_free_energy"],
                reference["radius"],
                reference["electrode_dist"],
            )

    @classmethod
    def generate(
        cls, entries: MappingDict
    ) -> Tuple[List[Reaction], Mapping_Family_Dict]:
        """
        A method to generate all the possible redox reactions from given entries

        Args:
            entries: ReactionNetwork(input_entries).entries,
               entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}

        Returns:
            list of RedoxReaction class objects
        """
        reactions = list()
        families = dict()
        for formula in entries:
            families[formula] = dict()
            for Nbonds in entries[formula]:
                charges = sorted(list(entries[formula][Nbonds].keys()))
                for charge in charges:
                    families[formula][charge] = list()
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
                                    if isomorphic:
                                        rct_mp, prdt_mp = generate_atom_mapping_1_1(
                                            node_mapping
                                        )
                                        r = cls(
                                            entry0,
                                            entry1,
                                            reactant_atom_mapping=rct_mp,
                                            product_atom_mapping=prdt_mp,
                                        )
                                        reactions.append(r)
                                        families[formula][charge0].append(r)

        return reactions, families

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of redox reaction (oxidation or reduction)

        Returns:
           Dictionary of the form
           {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B},
           where rnx_type_A is the primary type of the reaction based on the reactant
           and product of the RedoxReaction object, and the backwards of this reaction
           would be rnx_type_B.
        """

        if self.product.charge < self.reactant.charge:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {
            "class": "RedoxReaction",
            "rxn_type_A": rxn_type_A,
            "rxn_type_B": rxn_type_B,
        }
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
        A method to determine the free energy of the redox reaction. Note to
        set RedoxReaction.eletron_free_energy a value.

        Args:
           temperature:

        Returns:
            dictionary of the form
            {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}
            where free_energy_A is the primary type of the reaction based on the reactant
            and product of the RedoxReaction object, and the backwards of this reaction
            would be free_energy_B.
        """
        entry0 = self.reactant
        entry1 = self.product
        if entry1.get_free_energy() is not None and entry0.get_free_energy() is not None:
            free_energy_A = entry1.get_free_energy(temp=temperature) - entry0.get_free_energy(
                temp=temperature
            )
            free_energy_B = entry0.get_free_energy(temp=temperature) - entry1.get_free_energy(
                temp=temperature
            )

            if self.reaction_type()["rxn_type_A"] == "One electron reduction":
                free_energy_A += -self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B += -self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None
        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
        A method to determine the energy of the redox reaction

        Returns:
            Dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
            where energy_A is the primary type of the reaction based on the reactant and
            product of the RedoxReaction object, and the backwards of this reaction would
            be energy_B.
        """
        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy
        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, RedoxRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            free_energy = self.get_free_energy(temperature=temperature)

            if self.electrode_dist is None:
                kappa = 1
            else:
                kappa = np.exp(-1.2 * self.electrode_dist)

            if self.inner_reorganization_energy is None:
                delta_g_a = free_energy["free_energy_A"]
                delta_g_b = free_energy["free_energy_B"]
            else:
                lam_reorg = self.inner_reorganization_energy
                delta_g_a = (
                    lam_reorg / 4 * (1 + free_energy["free_energy_A"] / lam_reorg) ** 2
                )
                delta_g_b = (
                    lam_reorg / 4 * (1 + free_energy["free_energy_B"] / lam_reorg) ** 2
                )

            if (
                self.inner_reorganization_energy is None
                and free_energy["free_energy_A"] < 0
            ):
                rate_constant["k_A"] = kappa * k * temperature / h
            else:
                rate_constant["k_A"] = (
                    kappa
                    * k
                    * temperature
                    / h
                    * np.exp(-96487 * delta_g_a / (R * temperature))
                )

            if (
                self.inner_reorganization_energy is None
                and free_energy["free_energy_B"] < 0
            ):
                rate_constant["k_B"] = kappa * k * temperature / h
            else:
                rate_constant["k_B"] = (
                    kappa
                    * k
                    * temperature
                    / h
                    * np.exp(-96487 * delta_g_b / (R * temperature))
                )

            return rate_constant

    def as_dict(self) -> dict:
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
            "inner_reorganization_energy": self.inner_reorganization_energy,
            "dielectric": self.dielectric,
            "refractive": self.refractive,
            "electron_free_energy": self.electron_free_energy,
            "radius": self.radius,
            "electrode_dist": self.electrode_dist,
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
            d["electrode_dist"],
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
        reactant_atom_mapping: Atom_Mapping_Dict = None,
        product_atom_mapping: Atom_Mapping_Dict = None,
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

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a IntramolSingleBondChangeReaction class object into
        graph representation (nx.Digraph object).
        IntramolSingleBondChangeReaction must be of type 1 reactant -> 1 product

        Returns:
            nx.Digraph object of a single IntramolSingleBondChangeReaction object
        """

        return graph_rep_1_1(self)

    @classmethod
    def generate(
        cls, entries: MappingDict
    ) -> Tuple[List[Reaction], Mapping_Family_Dict]:
        reactions = list()
        families = dict()
        templates = list()
        for formula in entries:
            Nbonds_list = list(entries[formula].keys())
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
                        rxns, subgs = cls._generate_one(
                            entry1, entries, formula, Nbonds0, charge, cls
                        )
                        reactions.extend(rxns)
                        for r, g in zip(rxns, subgs):
                            families, templates = categorize(
                                r, families, templates, g, charge
                            )

        return reactions, families

    @staticmethod
    def _generate_one(
        entry1, entries, formula, Nbonds0, charge, cls
    ) -> Tuple[List[Reaction], List[nx.MultiDiGraph]]:
        """
        Helper function to generate reactions for one molecule entry.
        """
        reactions = []
        sub_graphs = []
        for bond in entry1.bonds:
            mg = copy.deepcopy(entry1.mol_graph)
            mg.break_edge(bond[0], bond[1], allow_reverse=True)
            if nx.is_weakly_connected(mg.graph):
                for entry0 in entries[formula][Nbonds0][charge]:
                    isomorphic, node_mapping = is_isomorphic(entry0.graph, mg.graph)
                    if isomorphic:
                        rct_mp, prdt_mp = generate_atom_mapping_1_1(node_mapping)
                        r = cls(
                            entry0,
                            entry1,
                            reactant_atom_mapping=rct_mp,
                            product_atom_mapping=prdt_mp,
                        )
                        indices = entry1.mol_graph.extract_bond_environment([bond])
                        subg = (
                            entry1.graph.subgraph(list(indices)).copy().to_undirected()
                        )

                        reactions.append(r)
                        sub_graphs.append(subg)

                        break

        return reactions, sub_graphs

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of intramolecular single bond change
        reaction (bond breakage or formation)

        Returns:
            Dictionary of the form {"class": "IntramolSingleBondChangeReaction",
            "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
            where rnx_type_A is the primary type of the reaction based on the
            reactant and product of the IntramolSingleBondChangeReaction
            object, and the backwards of this reaction would be rnx_type_B
        """
        if self.product.charge < self.reactant.charge:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        reaction_type = {
            "class": "IntramolSingleBondChangeReaction",
            "rxn_type_A": rxn_type_A,
            "rxn_type_B": rxn_type_B,
        }
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
        A method to determine the free energy of the intramolecular single bond change
        reaction.

        Args:
            temperature:

        Returns:
            Dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
            where free_energy_A is the primary type of the reaction based on
            the reactant and product of the IntramolSingleBondChangeReaction
            object, and the backwards of this reaction would be free_energy_B.
        """
        entry0 = self.reactant
        entry1 = self.product
        if entry1.get_free_energy() is not None and entry0.get_free_energy() is not None:
            free_energy_A = entry1.get_free_energy(temp=temperature) - entry0.get_free_energy(
                temp=temperature
            )
            free_energy_B = entry0.get_free_energy(temp=temperature) - entry1.get_free_energy(
                temp=temperature
            )
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
        A method to determine the energy of the intramolecular single bond change
        reaction.

        Returns:
            Dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
            where energy_A is the primary type of the reaction based on
            the reactant and product of the IntramolSingleBondChangeReaction object,
            and the backwards of this reaction would be energy_B.
        """

        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            free_energy = self.get_free_energy(temperature=temperature)

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = (
                    k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))
                )

            return rate_constant

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

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a IntermolecularReaction class object into graph
        representation (nx.Digraph object).
        IntermolecularReaction must be of type 1 reactant -> 2 products

        Returns:
            nx.Digraph object of a single IntermolecularReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(
        cls, entries: MappingDict
    ) -> Tuple[List[Reaction], Mapping_Family_Dict]:
        reactions = list()
        families = dict()
        templates = list()

        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds <= 0:
                    continue

                for charge in entries[formula][Nbonds]:
                    for entry in entries[formula][Nbonds][charge]:
                        rxns, subgs = cls._generate_one(entry, entries, charge, cls)
                        reactions.extend(rxns)
                        for r, g in zip(rxns, subgs):
                            families, templates = categorize(
                                r, families, templates, g, charge
                            )

        return reactions, families

    @staticmethod
    def _generate_one(
        entry, entries, charge, cls
    ) -> Tuple[List[Reaction], List[nx.MultiDiGraph]]:
        """
        Helper function to generate reactions for one molecule entry.
        """
        reactions = []
        sub_graphs = []

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
                                if isomorphic1:
                                    rct_mp, prdts_mp = generate_atom_mapping_1_2(
                                        entry, [entry0, entry1], [edge]
                                    )
                                    r = cls(
                                        entry,
                                        [entry0, entry1],
                                        reactant_atom_mapping=rct_mp,
                                        products_atom_mapping=prdts_mp,
                                    )

                                    mg = entry.mol_graph
                                    indices = mg.extract_bond_environment([edge])
                                    subg = (
                                        mg.graph.subgraph(list(indices))
                                        .copy()
                                        .to_undirected()
                                    )

                                    reactions.append(r)
                                    sub_graphs.append(subg)

                                    break
                            break
            except MolGraphSplitError:
                pass

        return reactions, sub_graphs

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of intermoleular reaction (bond
        decomposition from one to two or formation from two to one molecules)

        Returns:
            Dictionary of the form {"class": "IntermolecularReaction",
            "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B},
            where rnx_type_A is the primary type of the reaction based on the
            reactant and product of the IntermolecularReaction
            object, and the backwards of this reaction would be rnx_type_B
        """
        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {
            "class": "IntermolecularReaction",
            "rxn_type_A": rxn_type_A,
            "rxn_type_B": rxn_type_B,
        }
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
        A method to determine the free energy of the intermolecular reaction.

        Args:
            temperature:

        Returns:
            Dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
            where free_energy_A is the primary type of the reaction based on
            the reactant and product of the IntermolecularReaction
            object, and the backwards of this reaction would be free_energy_B.
        """
        g_entry = self.reactant.get_free_energy
        g_0 = self.product_0.get_free_energy
        g_1 = self.product_1.get_free_energy

        if g_1() is not None and g_0() is not None and g_entry() is not None:
            free_energy_A = (
                g_0(temp=temperature)
                + g_1(temp=temperature)
                - g_entry(temp=temperature)
            )
            free_energy_B = (
                g_entry(temp=temperature)
                - g_0(temp=temperature)
                - g_1(temp=temperature)
            )

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
        A method to determine the energy of the intermolecular reaction.

        Returns:
            Dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
            where energy_A is the primary type of the reaction based on the
            reactant and product of the IntermolecularReaction
            object, and the backwards of this reaction would be energy_B.
        """
        if (
            self.product_1.energy is not None
            and self.product_0.energy is not None
            and self.reactant.energy is not None
        ):
            energy_A = (
                self.product_0.energy + self.product_1.energy - self.reactant.energy
            )
            energy_B = (
                self.reactant.energy - self.product_0.energy - self.product_1.energy
            )

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            free_energy = self.get_free_energy(temperature=temperature)

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = (
                    k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))
                )

            return rate_constant

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

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a CoordinationBondChangeReaction class object into graph
        representation (nx.Digraph object).

        CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

        Returns:
             nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(
        cls, entries: MappingDict
    ) -> Tuple[List[Reaction], Mapping_Family_Dict]:

        # find metal entries
        M_entries = dict()
        for formula in entries:
            if formula in ["Li1", "Mg1", "Ca1", "Zn1"]:
                if formula not in M_entries:
                    M_entries[formula] = dict()
                for charge in entries[formula][0]:
                    assert len(entries[formula][0][charge]) == 1
                    M_entries[formula][charge] = entries[formula][0][charge][0]

        reactions = list()
        families = dict()
        templates = list()

        if not M_entries:
            return reactions, families

        for formula in entries:
            if "Li" in formula or "Mg" in formula or "Ca" in formula or "Zn" in formula:

                for Nbonds in entries[formula]:
                    if Nbonds <= 2:
                        continue

                    for charge in entries[formula][Nbonds]:
                        for entry in entries[formula][Nbonds][charge]:
                            rxns, subgs = cls._generate_one(
                                entry, entries, M_entries, cls
                            )
                            reactions.extend(rxns)
                            for r, g in zip(rxns, subgs):
                                families, templates = categorize(
                                    r, families, templates, g, charge
                                )

        return reactions, families

    @staticmethod
    def _generate_one(
        entry, entries, M_entries, cls
    ) -> Tuple[List[Reaction], List[nx.MultiDiGraph]]:
        """
        Helper function to generate reactions for one molecule entry.
        """
        reactions = []
        sub_graphs = []

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

        for bond_pair in bond_pairs:
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
                            if isomorphic:
                                this_m = M_entries[M_formula][M_charge]

                                rct_mp, prdts_mp = generate_atom_mapping_1_2(
                                    entry, [nonM_entry, this_m], bond_pair
                                )

                                r = cls(
                                    entry,
                                    [nonM_entry, this_m],
                                    reactant_atom_mapping=rct_mp,
                                    products_atom_mapping=prdts_mp,
                                )
                                mg = entry.mol_graph
                                indices = mg.extract_bond_environment(list(bond_pair))
                                subg = (
                                    mg.graph.subgraph(list(indices))
                                    .copy()
                                    .to_undirected()
                                )

                                reactions.append(r)
                                sub_graphs.append(subg)

                                break

            except MolGraphSplitError:
                pass

        return reactions, sub_graphs

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of coordination bond change reaction (bond breaking
        from one to two or forming from two to one molecules)

        Returns:
            Dictionary of the form {"class": "CoordinationBondChangeReaction",
                                    "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
            where rnx_type_A is the primary type of the reaction based on the
            reactant and product of the CoordinationBondChangeReaction
            object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {
            "class": "CoordinationBondChangeReaction",
            "rxn_type_A": rxn_type_A,
            "rxn_type_B": rxn_type_B,
        }
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
        A method to determine the free energy of the coordination bond change reaction

        Args:
            temperature:

        Returns:
            Dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
            where free_energy_A is the primary type of the reaction based
            on the reactant and product of the CoordinationBondChangeReaction
            object, and the backwards of this reaction would be free_energy_B.
        """
        g_entry = self.reactant.get_free_energy
        g_0 = self.product_0.get_free_energy
        g_1 = self.product_1.get_free_energy

        if g_1() is not None and g_0() is not None and g_entry() is not None:
            free_energy_A = (
                g_0(temp=temperature)
                + g_1(temp=temperature)
                - g_entry(temp=temperature)
            )
            free_energy_B = (
                g_entry(temp=temperature)
                - g_0(temp=temperature)
                - g_1(temp=temperature)
            )

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
        A method to determine the energy of the coordination bond change reaction

        Returns:
            Dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
            where energy_A is the primary type of the reaction based on the
            reactant and product of the CoordinationBondChangeReaction
            object, and the backwards of this reaction would be energy_B.
        """
        if (
            self.product_1.energy is not None
            and self.product_0.energy is not None
            and self.reactant.energy is not None
        ):
            energy_A = (
                self.product_0.energy + self.product_1.energy - self.reactant.energy
            )
            energy_B = (
                self.reactant.energy - self.product_0.energy - self.product_1.energy
            )

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            free_energy = self.get_free_energy()

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = (
                    k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))
                )

            return rate_constant

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
            reactant, product, transition_state=transition_state, parameters=parameters
        )

    def graph_representation(
        self,
    ) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

        """
        A method to convert a Concerted class object into graph
            representation (nx.Digraph object).
        IntermolecularReaction must be of type 1 reactant -> 2 products
        :return nx.Digraph object of a single IntermolecularReaction object
        """
        if len(self.reactants) == len(self.products) == 1:
            return graph_rep_1_1(self)
        elif len(self.reactants) == 1 and len(self.products) == 2:
            return graph_rep_1_2(self)
        elif len(self.reactants) == 2 and len(self.products) == 1:
            self.reactants, self.products = self.products, self.reactants
            return graph_rep_1_2(self)
        elif len(self.reactants) == len(self.products) == 2:
            return graph_rep_2_2(self)

    @classmethod
    def generate(
        cls,
        entries_list: [MoleculeEntry],
        name="nothing",
        read_file=False,
        num_processors=16,
        reaction_type="break2_form2",
        allowed_charge_change=0,
    ) -> Tuple[List[Reaction], Mapping_Family_Dict]:

        """
        A method to generate all the possible concerted reactions from given
        entries_list.
        Args:
           :param entries_list, entries_list = [MoleculeEntry]
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
        if read_file:
            all_concerted_reactions = loadfn(name + "_concerted_rxns.json")
        else:
            from pymatgen.reaction_network.extract_reactions import (
                FindConcertedReactions
            )

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

        # TODO: implement concept of reaction families for concerted reactions
        # with multiple reactants and multiple products
        return reactions, dict()

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        """
        A method to identify type of intermoleular reaction (bond decomposition
        from one to two or formation from two to one molecules)
        Args:
           :return dictionary of the form {"class": "IntermolecularReaction",
           "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
           where rnx_type_A is the primary type of the reaction based on the
           reactant and product of the IntermolecularReaction
           object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Concerted"
        rxn_type_B = "Concerted"

        reaction_type = {
            "class": "ConcertedReaction",
            "rxn_type_A": rxn_type_A,
            "rxn_type_B": rxn_type_B,
        }
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
        A method to determine the free energy of the concerted reaction
        Args:
           :return dictionary of the form {"free_energy_A": energy_A,
                                           "free_energy_B": energy_B}
           where free_energy_A is the primary type of the reaction based on
           the reactant and product of the ConcertedReaction
           object, and the backwards of this reaction would be free_energy_B.
        """

        if self.electron_free_energy is None:
            electron_free = 0.0
        else:
            electron_free = self.electron_free_energy

        cond_rct = all(
            reactant.get_free_energy() is not None for reactant in self.reactants
        )
        cond_pro = all(product.get_free_energy() is not None for product in self.products)
        if cond_rct and cond_pro:
            reactant_charge = np.sum([item.charge for item in self.reactants])
            product_charge = np.sum([item.charge for item in self.products])
            reactant_free_energy = np.sum(
                [item.get_free_energy(temp=temperature) for item in self.reactants]
            )
            product_free_energy = np.sum(
                [item.get_free_energy(temp=temperature) for item in self.products]
            )
            total_charge_change = product_charge - reactant_charge
            free_energy_A = (
                product_free_energy
                - reactant_free_energy
                + total_charge_change * electron_free
            )
            free_energy_B = (
                reactant_free_energy
                - product_free_energy
                - total_charge_change * electron_free
            )

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
        A method to determine the energy of the concerted reaction
        Args:
           :return dictionary of the form {"energy_A": energy_A,
                                           "energy_B": energy_B}
           where energy_A is the primary type of the reaction based on the
           reactant and product of the ConcertedReaction
           object, and the backwards of this reaction would be energy_B.
           Electron electronic energy set to 0 for now.
        """
        if all(reactant.energy is None for reactant in self.reactants) and all(
            product.energy is None for product in self.products
        ):
            reactant_total_charge = np.sum([item.charge for item in self.reactants])
            product_total_charge = np.sum([item.charge for item in self.products])
            reactant_total_energy = np.sum([item.energy for item in self.reactants])
            product_total_energy = np.sum([item.energy for item in self.products])
            # total_charge_change = product_total_charge - reactant_total_charge
            energy_A = product_total_energy - reactant_total_energy
            energy_B = reactant_total_energy - product_total_energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature
                ),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            free_energy = self.get_free_energy()

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = (
                    k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))
                )

            return rate_constant

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


def graph_rep_3_2(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 3 reactants -> 2
    products
    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction,
                        Concerted)
    """

    if len(reaction.reactants) != 3 or len(reaction.products) != 2:
        raise ValueError(
            "Must provide reaction with 3 reactants and 2 products for graph_rep_3_2"
        )

    reactant_0 = reaction.reactants[0]
    reactant_1 = reaction.reactants[1]
    reactant_2 = reaction.reactants[2]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_prod_name = (
            str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        )
        two_prod_name_entry_ids = (
            str(product_0.entry_id) + "+" + str(product_1.entry_id)
        )
    else:
        two_prod_name = (
            str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        )
        two_prod_name_entry_ids = (
            str(product_1.entry_id) + "+" + str(product_0.entry_id)
        )

    reactants_ind_list = np.array(
        [
            reactant_0.parameters["ind"],
            reactant_1.parameters["ind"],
            reactant_2.parameters["ind"],
        ]
    )
    reactant_inds = np.argsort(reactants_ind_list)
    reactants_ind_list = np.sort(reactants_ind_list)

    reactants_name = (
        str(reactants_ind_list[0])
        + "+"
        + str(reactants_ind_list[1])
        + "+"
        + str(reactants_ind_list[2])
    )
    reactants_name_entry_ids = (
        str(reactants_ind_list[reactant_inds[0]])
        + "+"
        + str(reactants_ind_list[reactant_inds[1]])
        + "+"
        + str(reactants_ind_list[reactant_inds[2]])
    )

    two_prod_name0 = (
        str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    )
    two_prod_name1 = (
        str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
    )

    if reactant_1.parameters["ind"] <= reactant_2.parameters["ind"]:
        three_reac_name0 = (
            str(reactant_0.parameters["ind"])
            + "+PR_"
            + str(reactant_1.parameters["ind"])
            + "+PR_"
            + str(reactant_2.parameters["ind"])
        )
        three_reac_entry_ids0 = (
            str(reactant_0.entry_id)
            + "+PR_"
            + str(reactant_1.entry_id)
            + "+PR_"
            + str(reactant_2.entry_id)
        )
    else:
        three_reac_name0 = (
            str(reactant_0.parameters["ind"])
            + "+PR_"
            + str(reactant_2.parameters["ind"])
            + "+PR_"
            + str(reactant_1.parameters["ind"])
        )
        three_reac_entry_ids0 = (
            str(reactant_0.entry_id)
            + "+PR_"
            + str(reactant_2.entry_id)
            + "+PR_"
            + str(reactant_1.entry_id)
        )
    if reactant_0.parameters["ind"] <= reactant_2.parameters["ind"]:
        three_reac_name1 = (
            str(reactant_1.parameters["ind"])
            + "+PR_"
            + str(reactant_0.parameters["ind"])
            + "+PR_"
            + str(reactant_2.parameters["ind"])
        )
        three_reac_entry_ids1 = (
            str(reactant_1.entry_id)
            + "+PR_"
            + str(reactant_0.entry_id)
            + "+PR_"
            + str(reactant_2.entry_id)
        )
    else:
        three_reac_name1 = (
            str(reactant_1.parameters["ind"])
            + "+PR_"
            + str(reactant_2.parameters["ind"])
            + "+PR_"
            + str(reactant_0.parameters["ind"])
        )
        three_reac_entry_ids1 = (
            str(reactant_1.entry_id)
            + "+PR_"
            + str(reactant_2.entry_id)
            + "+PR_"
            + str(reactant_0.entry_id)
        )
    if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
        three_reac_name2 = (
            str(reactant_2.parameters["ind"])
            + "+PR_"
            + str(reactant_0.parameters["ind"])
            + "+PR_"
            + str(reactant_1.parameters["ind"])
        )
        three_reac_entry_ids2 = (
            str(reactant_2.entry_id)
            + "+PR_"
            + str(reactant_0.entry_id)
            + "+PR_"
            + str(reactant_1.entry_id)
        )
    else:
        three_reac_name2 = (
            str(reactant_2.parameters["ind"])
            + "+PR_"
            + str(reactant_1.parameters["ind"])
            + "+PR_"
            + str(reactant_0.parameters["ind"])
        )
        three_reac_entry_ids2 = (
            str(reactant_2.entry_id)
            + "+PR_"
            + str(reactant_1.entry_id)
            + "+PR_"
            + str(reactant_0.entry_id)
        )

    node_name_A0 = three_reac_name0 + "," + two_prod_name
    node_name_A1 = three_reac_name1 + "," + two_prod_name
    node_name_A2 = three_reac_name2 + "," + two_prod_name
    node_name_B0 = two_prod_name0 + "," + reactants_name
    node_name_B1 = two_prod_name1 + "," + reactants_name

    two_prod_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_prod_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)

    entry_ids_name_A0 = three_reac_entry_ids0 + "," + two_prod_name_entry_ids
    entry_ids_name_A1 = three_reac_entry_ids1 + "," + two_prod_name_entry_ids
    entry_ids_name_A2 = three_reac_entry_ids2 + "," + two_prod_name_entry_ids
    entry_ids_name_B0 = two_prod_entry_ids0 + "," + reactants_name_entry_ids
    entry_ids_name_B1 = two_prod_entry_ids1 + "," + reactants_name_entry_ids

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(
        node_name_A0,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A0,
    )

    graph.add_edge(
        reactant_0.parameters["ind"],
        node_name_A0,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A0,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_A0,
        product_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_A1,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A1,
    )

    graph.add_edge(
        reactant_1.parameters["ind"],
        node_name_A1,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A1,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_A1,
        product_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_A2,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A2,
    )

    graph.add_edge(
        reactant_2.parameters["ind"],
        node_name_A2,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A1,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_A1,
        product_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_B0,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_name_B0,
    )

    graph.add_edge(
        product_0.parameters["ind"],
        node_name_B0,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_B0,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B0,
        reactant_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B0,
        reactant_2.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_B1,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_name_B1,
    )

    graph.add_edge(
        product_1.parameters["ind"],
        node_name_B1,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_B1,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B1,
        reactant_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B1,
        reactant_2.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        weight=1.0,
    )

    return graph


def graph_rep_2_2(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation.
    Reaction much be of type 2 reactants -> 2 products
    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction,
       IntramolSingleBondChangeReaction, Concerted)
    """

    if len(reaction.reactants) != 2 or len(reaction.products) != 2:
        raise ValueError(
            "Must provide reaction with 2 reactants and 2 products for graph_rep_2_2"
        )

    reactant_0 = reaction.reactants[0]
    reactant_1 = reaction.reactants[1]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_prod_name = (
            str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        )
        two_prod_name_entry_ids = (
            str(product_0.entry_id) + "+" + str(product_1.entry_id)
        )
    else:
        two_prod_name = (
            str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        )
        two_prod_name_entry_ids = (
            str(product_1.entry_id) + "+" + str(product_0.entry_id)
        )

    if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
        two_reac_name = (
            str(reactant_0.parameters["ind"]) + "+" + str(reactant_1.parameters["ind"])
        )
        two_reac_name_entry_ids = (
            str(reactant_0.entry_id) + "+" + str(reactant_1.entry_id)
        )
    else:
        two_reac_name = (
            str(reactant_1.parameters["ind"]) + "+" + str(reactant_0.parameters["ind"])
        )
        two_reac_name_entry_ids = (
            str(reactant_1.entry_id) + "+" + str(reactant_0.entry_id)
        )

    two_prod_name0 = (
        str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    )
    two_prod_name1 = (
        str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
    )

    two_reac_name0 = (
        str(reactant_0.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
    )
    two_reac_name1 = (
        str(reactant_1.parameters["ind"]) + "+PR_" + str(reactant_0.parameters["ind"])
    )

    node_name_A0 = two_reac_name0 + "," + two_prod_name
    node_name_A1 = two_reac_name1 + "," + two_prod_name
    node_name_B0 = two_prod_name0 + "," + two_reac_name
    node_name_B1 = two_prod_name1 + "," + two_reac_name

    two_prod_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_prod_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)

    two_reac_entry_ids0 = str(reactant_0.entry_id) + "+PR_" + str(reactant_1.entry_id)
    two_reac_entry_ids1 = str(reactant_1.entry_id) + "+PR_" + str(reactant_0.entry_id)

    entry_ids_name_A0 = two_reac_entry_ids0 + "," + two_prod_name_entry_ids
    entry_ids_name_A1 = two_reac_entry_ids1 + "," + two_prod_name_entry_ids
    entry_ids_name_B0 = two_prod_entry_ids0 + "," + two_reac_name_entry_ids
    entry_ids_name_B1 = two_prod_entry_ids1 + "," + two_reac_name_entry_ids

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(
        node_name_A0,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A0,
    )

    graph.add_edge(
        reactant_0.parameters["ind"],
        node_name_A0,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A0,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_A0,
        product_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_A1,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A1,
    )

    graph.add_edge(
        reactant_1.parameters["ind"],
        node_name_A1,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A1,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_A1,
        product_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_B0,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_name_B0,
    )

    graph.add_edge(
        product_0.parameters["ind"],
        node_name_B0,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )

    graph.add_edge(
        node_name_B0,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B0,
        reactant_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_B1,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_name_B1,
    )

    graph.add_edge(
        product_1.parameters["ind"],
        node_name_B1,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )

    graph.add_edge(
        node_name_B1,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B1,
        reactant_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )

    return graph


def graph_rep_1_2(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation.
    Reaction much be of type 1 reactant -> 2 products

    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction,
       IntramolSingleBondChangeReaction)
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 2:
        raise ValueError(
            "Must provide reaction with 1 reactant and 2 products" "for graph_rep_1_2"
        )

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_mol_name = (
            str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        )
        two_mol_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_mol_name = (
            str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        )
        two_mol_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    two_mol_name0 = (
        str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    )
    two_mol_name1 = (
        str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
    )
    node_name_A = str(reactant_0.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(reactant_0.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(reactant_0.parameters["ind"])

    two_mol_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_mol_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)
    entry_ids_name_A = str(reactant_0.entry_id) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(reactant_0.entry_id)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(reactant_0.entry_id)

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(
        node_name_A,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A,
    )

    graph.add_edge(
        reactant_0.parameters["ind"],
        node_name_A,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_A,
        product_1.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )

    graph.add_node(
        node_name_B0,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_name_B0,
    )
    graph.add_node(
        node_name_B1,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_name_B1,
    )

    graph.add_edge(
        node_name_B0,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_edge(
        node_name_B1,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )

    graph.add_edge(
        product_0.parameters["ind"],
        node_name_B0,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )
    graph.add_edge(
        product_1.parameters["ind"],
        node_name_B1,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )
    return graph


def graph_rep_1_1(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation.
    Reaction much be of type 1 reactant -> 1 product

    Args:
       :param reaction:(any of the reaction class object, ex. RedoxReaction,
       IntramolSingleBondChangeReaction)
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 1:
        raise ValueError(
            "Must provide reaction with 1 reactant and product" "for graph_rep_1_1"
        )

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    graph = nx.DiGraph()
    node_name_A = (
        str(reactant_0.parameters["ind"]) + "," + str(product_0.parameters["ind"])
    )
    node_name_B = (
        str(product_0.parameters["ind"]) + "," + str(reactant_0.parameters["ind"])
    )
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(reactant_0.entry_id) + "," + str(product_0.entry_id)
    entry_ids_B = str(product_0.entry_id) + "," + str(reactant_0.entry_id)

    graph.add_node(
        node_name_A,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_A,
    )
    graph.add_edge(
        reactant_0.parameters["ind"],
        node_name_A,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )
    graph.add_edge(
        node_name_A,
        product_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    graph.add_node(
        node_name_B,
        rxn_type=rxn_type_B,
        bipartite=1,
        energy=energy_B,
        free_energy=free_energy_B,
        entry_ids=entry_ids_B,
    )
    graph.add_edge(
        product_0.parameters["ind"],
        node_name_B,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )
    graph.add_edge(
        node_name_B,
        reactant_0.parameters["ind"],
        softplus=0.0,
        exponent=0.0,
        rexp=0.0,
        weight=1.0,
    )
    return graph


def categorize(reaction, families, templates, environment, charge):
    """
    Given reactants, products, and a local bonding environment, place a
        reaction into a reaction class.

    Note: This is not currently designed for redox reactions

    Args:
        reaction: Reaction object
        families: dict of dicts representing families of reactions
        templates: list of nx.Graph objects that define other families
        environment: a nx.Graph object representing a submolecule that
            defines the type of reaction
        charge: int representing the charge of the reaction
    Returns:
        families: nested dict containing categorized reactions
        templates: list of graph representations of molecule "templates"
    """

    nm = iso.categorical_node_match("specie", "ERROR")

    match = False
    bucket_templates = copy.deepcopy(templates)

    for e, template in enumerate(bucket_templates):
        if nx.is_isomorphic(environment, template, node_match=nm):
            match = True
            label = e
            if charge in families:
                if label in families[charge]:
                    families[charge][label].append(reaction)
                else:
                    families[charge][label] = [reaction]
                break
            else:
                families[charge] = {label: [reaction]}
                break
    if not match:
        label = len(templates)
        if charge in families:
            families[charge][label] = [reaction]
        else:
            families[charge] = {label: [reaction]}

        templates.append(environment)

    return families, templates


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


def generate_atom_mapping_1_1(
    node_mapping: Dict[int, int]
) -> Tuple[Atom_Mapping_Dict, Atom_Mapping_Dict]:
    """
    Generate rdkit style atom mapping for reactions with one reactant and one product.

    For example, given `node_mapping = {0:2, 1:0, 2:1}`, which means atoms 0, 1,
    and 2 in the reactant maps to atoms 2, 0, and 1 in the product, respectively,
    the atom mapping number for reactant atoms are simply set to their index,
    and the atom mapping number for product atoms are determined accordingly.
    As a result, this function gives: `({0:0, 1:1, 2:2}, {0:1 1:2 2:0})` as the output.
    Atoms in the reactant and product with the same atom mapping number
    (keys in the dicts) are corresponding to each other.

    Args:
        node_mapping: node mapping from reactant to product

    Returns:
        reactant_atom_mapping: rdkit style atom mapping for the reactant
        product_atom_mapping: rdkit style atom mapping for the product
    """
    reactant_atom_mapping = node_mapping
    product_atom_mapping = {v: k for k, v in node_mapping.items()}

    return reactant_atom_mapping, product_atom_mapping


def generate_atom_mapping_1_2(
    reactant: MoleculeEntry,
    products: List[MoleculeEntry],
    edges: List[Tuple[int, int]],
) -> Tuple[Atom_Mapping_Dict, List[Atom_Mapping_Dict]]:
    """
    Generate rdkit style atom mapping for reactions with one reactant and two products.

    The atom mapping number for reactant atoms are simply set to their index,
    and the atom mapping number for product atoms are determined accordingly.
    Atoms in the reactant and products with the same atom mapping number (value in the
    atom mapping dictionary {atom_index: atom_mapping_number}) corresponds to each other.

    For example, given reactant

          C 0
         / \
        /___\
       O     N---H
       1     2   3

    and the two products
          C 1
         / \
        /___\
       O     N
       2     0

    and
       H 0

    This function returns:
    reactant_atom_mapping = {0:0, 1:1, 2:2, 3:3}
    products_atom_mapping = [{0:2, 1:0, 2:1}, {0:3}]

    Args:
        reactant: reactant molecule entry
        products: products molecule entry
        edges: a list of bonds in reactant, by breaking which can form the two products

    Note:
        This function assumes the two subgraphs of the reactant obtained by breaking
        the edge are ordered the same as the products. i.e. subgraphs[0] corresponds to
        products[0] and subgraphs[1] corresponds to products[1].

    Returns:
        reactant_atom_mapping: rdkit style atom mapping number for the reactant
        products_atom_mapping: rdkit style atom mapping number for the two products
    """

    assert len(products) == 2, f"Expect 2 product molecules, got {len(products)}."

    reactant_atom_mapping = {i: i for i in range(reactant.num_atoms)}

    # Split the reactant mol graph to form two sub graphs
    # This is similar to MoleculeGraph.split_molecule_subbraphs(), but do not reorder
    # the nodes, i.e. the nodes in the subgraphs will have the same node indexes as
    original = copy.deepcopy(reactant.mol_graph)
    for edge in edges:
        original.break_edge(edge[0], edge[1], allow_reverse=True)
    components = nx.weakly_connected_components(original.graph)
    sub_graphs = [original.graph.subgraph(c) for c in components]

    products_atom_mapping = []
    for subg, prdt in zip(sub_graphs, products):
        _, node_mapping = is_isomorphic(prdt.graph, subg)
        assert node_mapping is not None, "Cannot obtain node mapping."
        products_atom_mapping.append(node_mapping)

    return reactant_atom_mapping, products_atom_mapping


def bucket_mol_entries(entries: List[MoleculeEntry], keys: Optional[List[str]] = None):
    """
    Bucket molecules into nested dictionaries according to molecule properties
    specified in keys.

    The nested dictionary has keys as given in `keys`, and the innermost value is a
    list. For example, if `keys = ['formula', 'Nbonds', 'charge']`, then the returned
    bucket dictionary is something like:

    bucket[formula][Nbonds][charge] = [mol_entry1, mol_entry2, ...]

    where mol_entry1, mol_entry2, ... have the same formula, number of bonds, and charge.

    Args:
        entries: a list of molecule entries to bucket
        keys: each str should be a molecule property.
            default to ['formula', 'Nbonds', 'charge']

    Returns:
        Nested dictionary of molecule entry bucketed according to keys.
    """
    keys = ["formula", "Nbonds", "charge"] if keys is None else keys

    num_keys = len(keys)
    buckets = {}
    for m in entries:
        b = buckets
        for i, k in enumerate(keys):
            v = getattr(m, k)
            if i == num_keys - 1:
                b.setdefault(v, []).append(m)
            else:
                b.setdefault(v, {})
            b = b[v]

    return buckets


def unbucket_mol_entries(entries: Dict) -> List[MoleculeEntry]:
    """
    Unbucket molecule entries stored in a nested dictionary to a list.

    This is the opposite operation to `bucket_mol_entries()`.

    Args:
        entries: nested dictionaries, e.g.
            bucket[formula][Nbonds][charge] = [mol_entry1, mol_entry2, ...]

    Returns:
        a list of molecule entries
    """

    def unbucket(d):
        for k, v in d.items():
            if isinstance(v, dict):
                unbucket(v)
            elif isinstance(v, Iterable):
                entries_list.extend(v)
            else:
                raise RuntimeError(
                    f"Cannot unbucket molecule entries. Unsupported data type `{type(v)}`"
                )

    entries_list = []
    unbucket(entries)

    return entries_list
