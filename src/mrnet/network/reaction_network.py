from abc import ABCMeta, abstractmethod
import copy
import itertools
import heapq
from typing import List, Dict, Tuple, Optional
import time as time
import numpy as np
from scipy.constants import h, k, R

import networkx as nx
from networkx.readwrite import json_graph
import networkx.algorithms.isomorphism as iso

from monty.json import MSONable
from monty.serialization import loadfn

from pymatgen.analysis.graphs import MolGraphSplitError
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.reaction_network.reaction_rates import (ReactionRateCalculator,
                                                      ExpandedBEPRateCalculator)
from pymatgen.util.classes import load_class


__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Energy_Dict = Dict[str, float]
Mapping_ReactionType_Dict = Dict[str, str]
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
       """

    def __init__(self, reactants: List[MoleculeEntry], products: List[MoleculeEntry],
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None):
        self.reactants = reactants
        self.products = products
        self.transition_state = transition_state
        if self.transition_state is None:
            # Provide no reference initially
            self.rate_calculator = None
        else:
            self.rate_calculator = ReactionRateCalculator(reactants, products,
                                                          self.transition_state)

        self.reactant_ids = [e.entry_id for e in self.reactants]
        self.product_ids = [e.entry_id for e in self.products]
        self.entry_ids = {e.entry_id for e in self.reactants}

        self.parameters = parameters or dict()

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactants)

    def update_calculator(self, transition_state: Optional[MoleculeEntry],
                          reference: Optional[Dict]):
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
            self.rate_calculator = ReactionRateCalculator(self.reactants, self.products,
                                                          transition_state)

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

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "transition_state": ts,
             "rate_calculator": rc,
             "parameters": self.parameters}

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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=d["parameters"])
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
       reactant([MoleculeEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry,
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None):
        """
            Initilizes RedoxReaction.reactant to be in the form of a MoleculeEntry,
            RedoxReaction.product to be in the form of MoleculeEntry,
            Reaction.reactant to be in the form of a of a list of MoleculeEntry of length 1
            Reaction.products to be in the form of a of a list of MoleculeEntry of length 1

          Args:
            reactant: MoleculeEntry object
            product: MoleculeEntry object
            transition_state (MoleculeEntry or None): A MoleculeEntry representing a
                transition state for the reaction.
            parameters (dict): Any additional data about this reaction

        """
        self.reactant = reactant
        self.product = product
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product],
                         transition_state=transition_state,
                         parameters=parameters)

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a RedoxReaction class object into graph representation (nx.Digraph object).
            Redox Reaction must be of type 1 reactant -> 1 product

            :return nx.Digraph object of a single Redox Reaction
        """

        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction],
                                                     Mapping_Family_Dict]:
        """
        A method to generate all the possible redox reactions from given entries

        Args:
           :param entries: ReactionNetwork(input_entries).entries,
               entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
           :return list of RedoxReaction class objects
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
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        r = cls(entry0, entry1)
                                        reactions.append(r)
                                        families[formula][charge0].append(r)

        return reactions, families

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of redox reaction (oxidation or reduction)

        Args:
           :return dictionary of the form {"class": "RedoxReaction",
                "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
           where rnx_type_A is the primary type of the reaction based on the
                reactant and product of the RedoxReaction
           object, and the backwards of this reaction would be rnx_type_B
        """

        if self.product.charge < self.reactant.charge:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {"class": "RedoxReaction",
                         "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
           A method to determine the free energy of the redox reaction. Note to
           set RedoxReaction.eletron_free_energy a value.

           Args:
              :return dictionary of the form {"free_energy_A": free_energy_A,
                                              "free_energy_B": free_energy_B}
              where free_energy_A is the primary type of the reaction based on
              the reactant and product of the RedoxReaction
              object, and the backwards of this reaction would be free_energy_B.
        """

        entry0 = self.reactant
        entry1 = self.product
        if entry1.free_energy() is not None and entry0.free_energy() is not None:
            free_energy_A = entry1.free_energy(temp=temperature) - entry0.free_energy(temp=temperature)
            free_energy_B = entry0.free_energy(temp=temperature) - entry1.free_energy(temp=temperature)

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

           Args:
              :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
              where energy_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
              object, and the backwards of this reaction would be energy_B.
        """
        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy
        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        rate_constant = dict()
        free_energy = self.free_energy(temperature=temperature)
        ea = 10000  # [J/mol] activation barrier for exothermic reactions
        if free_energy["free_energy_A"] < 0:
            rate_constant["k_A"] = k * temperature / h * np.exp(-1 * ea / (R * temperature))
        else:
            rate_constant["k_A"] = k * temperature / h * np.exp(-1 * free_energy["free_energy_A"] * 96487 /
                                                                (R * temperature))

        if free_energy["free_energy_B"] < 0:
            rate_constant["k_B"] = k * temperature / h * np.exp(-1 * ea / (R * temperature))
        else:
            rate_constant["k_B"] = k * temperature / h * np.exp(-1 * free_energy["free_energy_B"] * 96487 /
                                                                (R * temperature))

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

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product": self.product.as_dict(),
             "electron_free_energy": self.electron_free_energy,
             "transition_state": ts,
             "rate_calculator": rc,
             "parameters": self.parameters}

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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, product, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        reaction.electron_free_energy = d["electron_free_energy"]
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
       reactant([MoleculeEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
       transition_state (MoleculeEntry or None): A MoleculeEntry representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry,
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None):
        """
            Initilizes IntramolSingleBondChangeReaction.reactant to be in the form of a MoleculeEntry,
            IntramolSingleBondChangeReaction.product to be in the form of MoleculeEntry,
            Reaction.reactant to be in the form of a of a list of MoleculeEntry of length 1
            Reaction.products to be in the form of a of a list of MoleculeEntry of length 1

          Args:
            reactant: MoleculeEntry object
            product: MoleculeEntry object
            transition_state (MoleculeEntry or None): A MoleculeEntry representing a
                transition state for the reaction.
            parameters (dict): Any additional data about this reaction

        """

        self.reactant = reactant
        self.product = product
        super().__init__([self.reactant], [self.product],
                         transition_state=transition_state,
                         parameters=parameters)

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a IntramolSingleBondChangeReaction class object into
            graph representation (nx.Digraph object).
            IntramolSingleBondChangeReaction must be of type 1 reactant -> 1 product

            :return nx.Digraph object of a single IntramolSingleBondChangeReaction object
        """

        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction],
                                                     Mapping_Family_Dict]:
        reactions = list()
        families = dict()
        templates = list()
        for formula in entries:
            Nbonds_list = list(entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list) - 1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii + 1]
                    if Nbonds1 - Nbonds0 == 1:
                        for charge in entries[formula][Nbonds0]:
                            if charge not in families:
                                families[charge] = dict()
                            if charge in entries[formula][Nbonds1]:
                                for entry1 in entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0], edge[1], allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    r = cls(entry0, entry1)
                                                    reactions.append(r)
                                                    indices = entry1.mol_graph.extract_bond_environment([edge])
                                                    subg = entry1.graph.subgraph(list(indices)).copy().to_undirected()

                                                    families, templates = categorize(r, families, templates,
                                                                                     subg, charge)
                                                    break

        return reactions, families

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
            A method to identify type of intramolecular single bond change
            reaction (bond breakage or formation)

            Args:
               :return dictionary of the form {"class": "IntramolSingleBondChangeReaction",
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

        reaction_type = {"class": "IntramolSingleBondChangeReaction",
                         "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
          A method to  determine the free energy of the intramolecular single
          bond change reaction

          Args:
             :return dictionary of the form {"free_energy_A": energy_A,
                                             "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on
             the reactant and product of the IntramolSingleBondChangeReaction
             object, and the backwards of this reaction would be free_energy_B.
        """
        entry0 = self.reactant
        entry1 = self.product
        if entry1.free_energy() is not None and entry0.free_energy() is not None:
            free_energy_A = entry1.free_energy(temp=temperature) - entry0.free_energy(temp=temperature)
            free_energy_B = entry0.free_energy(temp=temperature) - entry1.free_energy(temp=temperature)
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the intramolecular single bond
          change reaction

          Args:
             :return dictionary of the form {"energy_A": energy_A,
                                             "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the
             reactant and product of the IntramolSingleBondChangeReaction
             object, and the backwards of this reaction would be energy_B.
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
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        else:
            rate_constant = dict()
            free_energy = self.free_energy(temperature=temperature)

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))

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

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product": self.product.as_dict(),
             "transition_state": ts,
             "rate_calculator": rc,
             "parameters": self.parameters}

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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, product, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class IntermolecularReaction(Reaction):
    """
        A class to define intermolecular bond change as follows:
            Intermolecular formation / breakage of one bond
            A <-> B + C aka B + C <-> A
            Three entries with:
                comp(A) = comp(B) + comp(C)
                charge(A) = charge(B) + charge(C)
                removing one of the edges in A yields two disconnected subgraphs
                that are isomorphic to B and C

        Args:
            reactant([MoleculeEntry]): list of single molecular entry
            product([MoleculeEntry]): list of two molecular entries
            transition_state (MoleculeEntry or None): A MoleculeEntry
                representing a
                transition state for the reaction.
            parameters (dict): Any additional data about this reaction
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry],
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None):
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
            transition_state (MoleculeEntry or None): A MoleculeEntry
                representing a transition state for the reaction.
            parameters (dict): Any additional data about this reaction

        """

        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1],
                         transition_state=transition_state,
                         parameters=parameters)

    def graph_representation(self) -> nx.DiGraph:

        """
            A method to convert a IntermolecularReaction class object into graph
            representation (nx.Digraph object).
            IntermolecularReaction must be of type 1 reactant -> 2 products

            :return nx.Digraph object of a single IntermolecularReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction],
                                                     Mapping_Family_Dict]:
        reactions = list()
        families = dict()
        templates = list()
        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds > 0:
                    for charge in entries[formula][Nbonds]:
                        if charge not in families:
                            families[charge] = dict()
                        for entry in entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0], edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond,
                                                                                     allow_reverse=True)
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in entries and formula1 in entries:
                                        if Nbonds0 in entries[formula0] and Nbonds1 in entries[formula1]:
                                            for charge0 in entries[formula0][Nbonds0]:
                                                for entry0 in entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in entries[formula1][Nbonds1]:
                                                            for entry1 in entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    r = cls(entry, [entry0, entry1])
                                                                    mg = entry.mol_graph
                                                                    indices = mg.extract_bond_environment([edge])
                                                                    subg = mg.graph.subgraph(list(indices)).copy()
                                                                    subg = subg.to_undirected()

                                                                    families, templates = categorize(r, families,
                                                                                                     templates, subg,
                                                                                                     charge)
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions, families

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        """
           A method to identify type of intermoleular reaction (bond
           decomposition from one to two or formation from two to one molecules)

           Args:
              :return dictionary of the form {"class": "IntermolecularReaction",
              "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the
              reactant and product of the IntermolecularReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {"class": "IntermolecularReaction",
                         "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
          A method to determine the free energy of the intermolecular reaction

          Args:
             :return dictionary of the form {"free_energy_A": energy_A,
                                             "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on
             the reactant and product of the IntermolecularReaction
             object, and the backwards of this reaction would be free_energy_B.
         """
        g_entry = self.reactant.free_energy
        g_0 = self.product_0.free_energy
        g_1 = self.product_1.free_energy

        if g_1() is not None and g_0() is not None and g_entry() is not None:
            free_energy_A = g_0(temp=temperature) + g_1(temp=temperature) - g_entry(temp=temperature)
            free_energy_B = g_entry(temp=temperature) - g_0(temp=temperature) - g_1(temp=temperature)

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the intermolecular reaction

          Args:
             :return dictionary of the form {"energy_A": energy_A,
                                             "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the
             reactant and product of the IntermolecularReaction
             object, and the backwards of this reaction would be energy_B.
        """
        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        else:
            rate_constant = dict()
            free_energy = self.free_energy(temperature=temperature)

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))

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

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product_0": self.product_0.as_dict(),
             "product_1": self.product_1.as_dict(),
             "transition_state": ts,
             "rate_calculator": rc,
             "parameters": self.parameters}

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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, [product_0, product_1], transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class CoordinationBondChangeReaction(Reaction):
    """
    A class to define coordination bond change as follows:
        Simultaneous formation / breakage of multiple coordination bonds
        A + M <-> AM aka AM <-> A + M
        Three entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            charge(AM) = charge(A) + charge(M)
            removing two M-containing edges in AM yields two disconnected
            subgraphs that are isomorphic to B and C

    Args:
        reactant([MoleculeEntry]): list of single molecular entry
        product([MoleculeEntry]): list of two molecular entries
        transition_state (MoleculeEntry or None): A MoleculeEntry representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry],
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None):
        """
            Initilizes CoordinationBondChangeReaction.reactant to be in the
            form of a MoleculeEntry,
            CoordinationBondChangeReaction.product to be in the form of
                [MoleculeEntry_0, MoleculeEntry_1],
            Reaction.reactant to be in the form of a of a list of MoleculeEntry
                of length 1
            Reaction.products to be in the form of a of a list of MoleculeEntry
                of length 2

        Args:
            reactant: MoleculeEntry object
            product: list of MoleculeEntry object of length 2
            transition_state (MoleculeEntry or None): A MoleculeEntry
                representing a transition state for the reaction.
            parameters (dict): Any additional data about this reaction

        """
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1],
                         transition_state=transition_state,
                         parameters=parameters)

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a CoordinationBondChangeReaction class object
                into graph representation (nx.Digraph object).
            CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

            :return nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction],
                                                     Mapping_Family_Dict]:
        reactions = list()
        M_entries = dict()
        families = dict()
        fam = dict()
        temp = list()
        for formula in entries:
            if formula in ["Li1", "Mg1", "Ca1", "Zn1"]:
                if formula not in M_entries:
                    M_entries[formula] = dict()
                for charge in entries[formula][0]:
                    assert (len(entries[formula][0][charge]) == 1)
                    M_entries[formula][charge] = entries[formula][0][charge][0]
        if M_entries != dict():
            for formula in entries:
                if "Li" in formula or "Mg" in formula or "Ca" in formula or "Zn" in formula:
                    for Nbonds in entries[formula]:
                        if Nbonds > 2:
                            for charge in entries[formula][Nbonds]:
                                if charge not in families:
                                    families[charge] = dict()
                                for entry in entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = list()
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(
                                                entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0], edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond],
                                                                                                 allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair,
                                                                                             allow_reverse=True)
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
                                                        nonM_formula = frag.molecule.composition.alphabetical_formula
                                                        nonM_Nbonds = len(frag.graph.edges())
                                                        if nonM_formula in entries:
                                                            if nonM_Nbonds in entries[nonM_formula]:
                                                                for nonM_charge in entries[nonM_formula][nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in \
                                                                                entries[nonM_formula][nonM_Nbonds][
                                                                                    nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                this_m = M_entries[M_formula][M_charge]
                                                                                r = cls(entry, [nonM_entry, this_m])

                                                                                mg = entry.mol_graph

                                                                                indices = mg.extract_bond_environment(list(bond_pair))
                                                                                subg = mg.graph.subgraph(list(indices)).copy().to_undirected()

                                                                                fam, temp = categorize(r,
                                                                                                       fam,
                                                                                                       temp,
                                                                                                       subg,
                                                                                                       charge)

                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions, fam

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
           A method to identify type of coordination bond change reaction
           (bond breaking from one to two or forming from two to one molecules)

           Args:
              :return dictionary of the form {"class": "CoordinationBondChangeReaction",
                                              "rxn_type_A": rxn_type_A,
                                              "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the
              reactant and product of the CoordinationBondChangeReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {"class": "CoordinationBondChangeReaction",
                         "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
              A method to determine the free energy of the coordination bond
                change reaction

              Args:
                 :return dictionary of the form {"free_energy_A": energy_A,
                                                 "free_energy_B": energy_B}
                 where free_energy_A is the primary type of the reaction based
                 on the reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be free_energy_B.
         """
        g_entry = self.reactant.free_energy
        g_0 = self.product_0.free_energy
        g_1 = self.product_1.free_energy

        if g_1() is not None and g_0() is not None and g_entry() is not None:
            free_energy_A = g_0(temp=temperature) + g_1(temp=temperature) - g_entry(temp=temperature)
            free_energy_B = g_entry(temp=temperature) - g_0(temp=temperature) - g_1(temp=temperature)

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
              A method to determine the energy of the coordination bond change
              reaction

              Args:
                 :return dictionary of the form {"energy_A": energy_A,
                                                 "energy_B": energy_B}
                 where energy_A is the primary type of the reaction based on the
                 reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be energy_B.
        """
        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        else:
            rate_constant = dict()
            free_energy = self.free_energy()

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))

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

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "reactant": self.reactant.as_dict(),
             "product_0": self.product_0.as_dict(),
             "product_1": self.product_1.as_dict(),
             "transition_state": ts,
             "rate_calculator": rc,
             "parameters": self.parameters}

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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, [product_0, product_1], transition_state=ts,
                       parameters=parameters)
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

    def __init__(self, reactant: List[MoleculeEntry], product: List[MoleculeEntry],
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None):
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
        """

        self.reactants = reactant
        self.products = product
        self.electron_free_energy = None
        self.electron_energy = None
        super().__init__(reactant, product,
                         transition_state=transition_state,
                         parameters=parameters)

    def graph_representation(self) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

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
    def generate(cls, entries_list: [MoleculeEntry], name="nothing",
                 read_file=False, num_processors=16,
                 reaction_type="break2_form2", allowed_charge_change=0) -> Tuple[List[Reaction],
                                                                                 Mapping_Family_Dict]:

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
            all_concerted_reactions = loadfn(name+'_concerted_rxns.json')
        else:
            from pymatgen.reaction_network.extract_reactions import FindConcertedReactions
            FCR = FindConcertedReactions(entries_list, name)
            all_concerted_reactions = FCR.get_final_concerted_reactions(name,
                                                                        num_processors,
                                                                        reaction_type)

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

        reaction_type = {"class": "ConcertedReaction",
                         "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
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
        cond_rct = all(reactant.free_energy() is not None for reactant in self.reactants)
        cond_pro = all(product.free_energy() is not None for product in self.products)
        if cond_rct and cond_pro:
            reactant_charge = np.sum([item.charge for item in self.reactants])
            product_charge = np.sum([item.charge for item in self.products])
            reactant_free_energy = np.sum([item.free_energy(temp=temperature) for item in self.reactants])
            product_free_energy = np.sum([item.free_energy(temp=temperature) for item in self.products])
            total_charge_change = product_charge - reactant_charge
            free_energy_A = product_free_energy - reactant_free_energy + total_charge_change * self.electron_free_energy
            free_energy_B = reactant_free_energy - product_free_energy - total_charge_change * self.electron_free_energy

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
                product.energy is None for product in self.products):
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
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {"k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                    "k_B": self.rate_calculator.calculate_rate_constant(temperature=temperature,
                                                                        reverse=True)}
        else:
            rate_constant = dict()
            free_energy = self.free_energy()

            ga = free_energy["free_energy_A"]
            gb = free_energy["free_energy_B"]

            if free_energy["free_energy_A"] < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))

            if free_energy["free_energy_B"] < 0:
                rate_constant["k_B"] = k * temperature / h
            else:
                rate_constant["k_B"] = k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))

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

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "transition_state": ts,
             "rate_calculator": rc,
             "parameters": self.parameters}

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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactants, products, transition_state=ts,
                       parameters=parameters)
        reaction.rate_calculator = rate_calculator
        return reaction


class ReactionPath(MSONable):
    """
        A class to define path object within the reaction network which
        constains all the associated characteristic attributes of a given path

        :param path - a list of nodes that defines a path from node A to B
            within a graph built using ReactionNetwork.build()
    """

    def __init__(self, path):
        """
        initializes the ReactionPath object attributes for a given path
        :param path: a list of nodes that defines a path from node A to B
            within a graph built using ReactionNetwork.build()
        """

        self.path = path
        self.byproducts = []
        self.unsolved_prereqs = []
        self.solved_prereqs = []
        self.all_prereqs = []
        self.cost = 0.0
        self.overall_free_energy_change = 0.0
        self.hardest_step = None
        self.description = ""
        self.pure_cost = 0.0
        self.full_path = None
        self.hardest_step_deltaG = None
        self.path_dict = {"byproducts": self.byproducts, "unsolved_prereqs": self.unsolved_prereqs,
                          "solved_prereqs": self.solved_prereqs, "all_prereqs": self.all_prereqs, "cost": self.cost,
                          "path": self.path, "overall_free_energy_change": self.overall_free_energy_change,
                          "hardest_step": self.hardest_step, "description": self.description,
                          "pure_cost": self.pure_cost,
                          "hardest_step_deltaG": self.hardest_step_deltaG, "full_path": self.full_path}

    @property
    def as_dict(self) -> dict:
        """
            A method to convert ReactionPath objection into a dictionary
        :return: d: dictionary containing all te ReactionPath attributes
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "byproducts": self.byproducts,
             "unsolved_prereqs": self.unsolved_prereqs,
             "solved_prereqs": self.solved_prereqs,
             "all_prereqs": self.all_prereqs,
             "cost": self.cost,
             "path": self.path,
             "overall_free_energy_change": self.overall_free_energy_change,
             "hardest_step": self.hardest_step,
             "description": self.description,
             "pure_cost": self.pure_cost,
             "hardest_step_deltaG": self.hardest_step_deltaG,
             "full_path": self.full_path,
             "path_dict": self.path_dict
             }
        return d

    @classmethod
    def from_dict(cls, d):
        """
            A method to convert dict to ReactionPath object
        :param d:  dict retuend from ReactionPath.as_dict() method
        :return: ReactionPath object
        """
        x = cls(d.get("path"))
        x.byproducts = d.get("byproducts")
        x.unsolved_prereqs = d.get("unsolved_prereqs")
        x.solved_prereqs = d.get("solved_prereqs")
        x.all_prereqs = d.get("all_prereqs")
        x.cost = d.get("cost", 0)

        x.overall_free_energy_change = d.get("overall_free_energy_change", 0)
        x.hardest_step = d.get("hardest_step")
        x.description = d.get("description")
        x.pure_cost = d.get("pure_cost", 0)
        x.hardest_step_deltaG = d.get("hardest_step_deltaG")
        x.full_path = d.get("full_path")
        x.path_dict = d.get("path_dict")

        return x

    @classmethod
    def characterize_path(cls, path: List[str], weight: str,
                          min_cost: Dict[str, float], graph: nx.DiGraph,
                          old_solved_PRs=[], PR_byproduct_dict={},
                          actualPRs={}):  # -> ReactionPath
        """
            A method to define ReactionPath attributes based on the inputs

        :param path: a list of nodes that defines a path from node A to B
            within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of
            from {node: float}
        :param graph: nx.Digraph
        :param old_solved_PRs: previously solved PRs from the iterations before
            the current iteration
        :param PR_byproduct_dict: dict of solved PR and its list of byproducts
        :param actualPRs: PR dictionary
        :return: ReactionPath object
        """

        if path is None:
            class_instance = cls(None)
        else:
            class_instance = cls(path)
            pool = []
            pool.append(path[0])
            for ii, step in enumerate(path):
                if ii != len(path) - 1:
                    class_instance.cost += graph[step][path[ii + 1]][weight]
                    if ii % 2 == 1:
                        rxn = step.split(",")
                        if "+PR_" in rxn[0]:
                            a = int(rxn[0].split("+PR_")[0])
                            PR_b = int(rxn[0].split("+PR_")[1])
                            concerted = False
                            PR_b2 = None
                            if rxn[0].count("PR_") == 2:
                                PR_b2 = int(rxn[0].split("+PR_")[2])
                            if "+" in rxn[1]:
                                concerted = True
                                c = int(rxn[1].split("+")[0])
                                d = int(rxn[1].split("+")[1])
                            else:
                                c = int(rxn[1])
                            pool_modified = copy.deepcopy(pool)
                            pool_modified.remove(a)
                            if PR_b2 == None:
                                if PR_b in pool_modified:
                                    if PR_b in list(min_cost.keys()):
                                        class_instance.cost = class_instance.cost - min_cost[PR_b]
                                    else:
                                        pass
                                    pool.remove(a)
                                    pool.remove(PR_b)
                                    pool.append(c)
                                    if concerted:
                                        pool.append(d)
                                elif PR_b not in pool_modified:
                                    if PR_b in old_solved_PRs:
                                        class_instance.solved_prereqs.append(PR_b)
                                        class_instance.all_prereqs.append(PR_b)
                                        PR_b_byproducts = PR_byproduct_dict[PR_b]["byproducts"]
                                        start = int(PR_byproduct_dict[PR_b]["start"])
                                        if a in PR_b_byproducts:
                                            #print("path replacement happenning")
                                            new_path_piece1 = actualPRs[PR_b][start].path
                                            new_path_piece2 = [str(PR_b) + "+" + "PR_" + str(a) + "," + str(c)]
                                            if concerted:
                                                new_path_piece2 = [str(PR_b) + "+" + "PR_" + str(a) + "," + str(c)+"+"+str(d)]
                                            new_path_piece3 = path[ii + 1::]
                                            new_path = new_path_piece1 + new_path_piece2 + new_path_piece3
                                            #print(path, new_path_piece1, new_path_piece2,new_path_piece3 )
                                            assert (c == path[ii + 1] or d == path[ii + 1])
                                            if new_path_piece2[0] not in graph.nodes:
                                                pool.remove(a)
                                                pool = pool + PR_b_byproducts
                                                pool.append(c)
                                                if concerted:
                                                    pool.append(d)
                                            else:
                                                return ReactionPath.characterize_path(new_path, weight, min_cost, graph,
                                                                                  old_solved_PRs, PR_byproduct_dict,
                                                                                  actualPRs)
                                        elif a not in PR_b_byproducts:
                                            pool.remove(a)
                                            pool = pool + PR_b_byproducts
                                            pool.append(c)
                                            if concerted:
                                                pool.append(d)
                                    elif PR_b not in old_solved_PRs:
                                        class_instance.unsolved_prereqs.append(PR_b)
                                        class_instance.all_prereqs.append(PR_b)
                                        pool.remove(a)
                                        pool.append(c)
                                        if concerted:
                                            pool.append(d)
                            else:  # nodes with 2 PRs
                                if PR_b in pool_modified and PR_b2 in pool_modified:
                                    # print("!!")
                                    class_instance.cost = class_instance.cost - min_cost[PR_b]
                                    class_instance.cost = class_instance.cost - min_cost[PR_b2]
                                    pool.remove(a)
                                    pool.remove(PR_b)
                                    pool.remove(PR_b2)
                                    pool.append(c)
                                    pool.append(d)

                                elif PR_b not in old_solved_PRs and PR_b2 not in old_solved_PRs:
                                    class_instance.unsolved_prereqs.append(PR_b)
                                    class_instance.unsolved_prereqs.append(PR_b2)
                                    class_instance.all_prereqs.append(PR_b)
                                    class_instance.all_prereqs.append(PR_b2)
                                    pool.remove(a)
                                    pool.append(c)
                                    pool.append(d)


                                elif PR_b not in pool_modified and PR_b2 not in pool_modified:
                                    if PR_b in old_solved_PRs and PR_b2 in old_solved_PRs:
                                        pool.remove(a)
                                        pool.append(c)
                                        pool.append(d)
                                        class_instance.solved_prereqs.append(PR_b)
                                        class_instance.solved_prereqs.append(PR_b2)
                                        class_instance.all_prereqs.append(PR_b)
                                        class_instance.all_prereqs.append(PR_b2)
                                        PR_b_byproducts = PR_byproduct_dict[PR_b]["byproducts"]
                                        PR_b2_byproducts = PR_byproduct_dict[PR_b2]["byproducts"]
                                        pool = pool + PR_b_byproducts + PR_b2_byproducts

                                    elif PR_b not in old_solved_PRs or PR_b2 not in old_solved_PRs:
                                        if PR_b not in old_solved_PRs:
                                            class_instance.unsolved_prereqs.append(PR_b)
                                            class_instance.all_prereqs.append(PR_b)
                                        elif PR_b2 not in old_solved_PRs:
                                            class_instance.unsolved_prereqs.append(PR_b2)
                                            class_instance.all_prereqs.append(PR_b2)
                                        pool.remove(a)
                                        pool.append(c)
                                        pool.append(d)


                                elif PR_b in pool_modified or PR_b2 in pool_modified:
                                    # print("$$")
                                    if PR_b in pool_modified:
                                        PR_in_pool = PR_b
                                        PR_not_in_pool = PR_b2
                                    elif PR_b2 in pool_modified:
                                        PR_in_pool = PR_b2
                                        PR_not_in_pool = PR_b
                                    if PR_not_in_pool in old_solved_PRs:
                                        class_instance.cost = class_instance.cost - min_cost[PR_in_pool]
                                        pool.remove(PR_in_pool)

                                    elif PR_not_in_pool in old_solved_PRs:
                                        class_instance.unsolved_prereqs.append(PR_not_in_pool)
                                        class_instance.all_prereqs.append(PR_not_in_pool)
                                    pool.remove(a)
                                    pool.append(c)
                                    pool.append(d)

                        elif "+" in rxn[1]:
                            # node = A,B+C
                            a = int(rxn[0])
                            b = int(rxn[1].split("+")[0])
                            c = int(rxn[1].split("+")[1])
                            pool.remove(a)
                            pool.append(b)
                            pool.append(c)
                        else:
                            # node = A,B
                            a = int(rxn[0])
                            b = int(rxn[1])
                            pool.remove(a)
                            pool.append(b)
            pool.remove(path[-1])
            class_instance.byproducts = pool

            class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                        "unsolved_prereqs": class_instance.unsolved_prereqs,
                                        "solved_prereqs": class_instance.solved_prereqs,
                                        "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                        "path": class_instance.path,
                                        "overall_free_energy_change": class_instance.overall_free_energy_change,
                                        "hardest_step": class_instance.hardest_step,
                                        "description": class_instance.description,
                                        "pure_cost": class_instance.pure_cost,
                                        "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                        "full_path": class_instance.full_path}

        return class_instance
    @classmethod
    def characterize_path_final(cls, path: List[str], weight: str,
                                min_cost: Dict[str, float], graph: nx.DiGraph,
                                old_solved_PRs=[], PR_byproduct_dict={},
                                PR_paths={}):
        """
            A method to define all the attributes of a given path once all the
            PRs are solved

        :param path: a list of nodes that defines a path from node A to B
            within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of
            from {node: float}, if no path exist, value is "no_path", if path is
            unsolved yet, value is "unsolved_path"
        :param graph: nx.Digraph
        :param old_solved_PRs: previously solved PRs from the iterations before
            the current iteration
        :param PR_byproduct_dict: dict of solved PR and its list of byproducts
        :param PR_paths: dict that defines a path from each node to a start,
               of the form {int(node1): {int(start1}: {ReactionPath object},
               int(start2): {ReactionPath object}}, int(node2):...}
        :return: ReactionPath object
        """

        if path is None:
            class_instance = cls(None)
        else:
            class_instance = cls.characterize_path(path, weight, min_cost, graph, old_solved_PRs, PR_byproduct_dict,
                                                   PR_paths)
            assert (len(class_instance.solved_prereqs) == len(class_instance.all_prereqs))
            assert (len(class_instance.unsolved_prereqs) == 0)

            PRs_to_join = copy.deepcopy(class_instance.all_prereqs)
            full_path = copy.deepcopy(path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = None
                    PR_min_cost = float("inf")  # 1000000000000000.0
                    for start in PR_paths[PR]:
                        if PR_paths[PR][start].path != None:
                            if PR_paths[PR][start].cost < PR_min_cost:
                                PR_min_cost = PR_paths[PR][start].cost
                                PR_path = PR_paths[PR][start]
                    assert (len(PR_path.solved_prereqs) == len(PR_path.all_prereqs))
                    for new_PR in PR_path.all_prereqs:
                        new_PRs.append(new_PR)
                    full_path = PR_path.path + full_path
                PRs_to_join = copy.deepcopy(new_PRs)

            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    print("WARNING: Matching prereq and byproduct found!", PR)

            for ii, step in enumerate(full_path):
                if graph.nodes[step]["bipartite"] == 1:
                    if weight == "softplus":
                        class_instance.pure_cost += ReactionNetwork.softplus(graph.nodes[step]["free_energy"])
                    elif weight == "exponent":
                        class_instance.pure_cost += ReactionNetwork.exponent(graph.nodes[step]["free_energy"])
                    elif weight == "rexp":
                        class_instance.pure_cost += ReactionNetwork.rexp(graph.nodes[step]["free_energy"])

                    class_instance.overall_free_energy_change += graph.nodes[step]["free_energy"]

                    if class_instance.description == "":
                        class_instance.description += graph.nodes[step]["rxn_type"]
                    else:
                        class_instance.description += ", " + graph.nodes[step]["rxn_type"]

                    if class_instance.hardest_step is None:
                        class_instance.hardest_step = step
                    elif graph.nodes[step]["free_energy"] > graph.nodes[class_instance.hardest_step]["free_energy"]:
                        class_instance.hardest_step = step

            class_instance.full_path = full_path

            if class_instance.hardest_step is None:
                class_instance.hardest_step_deltaG = None
            else:
                class_instance.hardest_step_deltaG = graph.nodes[class_instance.hardest_step]["free_energy"]

        class_instance.just_path_bp = []
        for ii, step in enumerate(class_instance.path):
            if isinstance(step, int):
                pass
            elif graph.nodes[step]["rxn_type"] == "Molecular decomposition breaking one bond A -> B+C":
                prods = step.split(",")[1].split("+")
                for p in prods:
                    if int(class_instance.path[ii + 1]) != int(p):
                        class_instance.just_path_bp.append(int(p))

        class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                    "just_path_bp": class_instance.just_path_bp,
                                    "unsolved_prereqs": class_instance.unsolved_prereqs,
                                    "solved_prereqs": class_instance.solved_prereqs,
                                    "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                    "path": class_instance.path,
                                    "overall_free_energy_change": class_instance.overall_free_energy_change,
                                    "hardest_step": class_instance.hardest_step,
                                    "description": class_instance.description, "pure_cost": class_instance.pure_cost,
                                    "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                    "full_path": class_instance.full_path}

        return class_instance

Mapping_PR_Dict = Dict[int, Dict[int, ReactionPath]]


class ReactionNetwork(MSONable):
    """
       Class to build a reaction network from entries

    """

    def __init__(self, electron_free_energy, temperature, entries_dict,
                 entries_list, graph, reactions, families,
                 PR_record, min_cost, num_starts):
        """
        :param electron_free_energy: Electron free energy (in eV)
        :param temperature: Temperature of the system, used for free energy
            and rate constants (temperature given in K)
        :param entries_dict: dict of dicts of dicts of lists (d[formula][bonds][charge])
        :param entries_list: list of unique entries in entries_dict
        :param graph: nx.DiGraph representing connections in the network
        :param reactions: list of Reaction objects
        :param families: dict containing reaction families
        :param PR_record: dict containing reaction prerequisites
        :param min_cost: dict containing costs of entries in the network
        :param num_starts: Number of starting molecules
        """

        self.electron_free_energy = electron_free_energy
        self.temperature = temperature

        self.entries = entries_dict
        self.entries_list = entries_list

        self.graph = graph
        self.PR_record = PR_record
        self.reactions = reactions
        self.families = families

        self.min_cost = min_cost
        self.num_starts = num_starts
        self.PRs = {}

        self.reachable_nodes = []
        self.unsolvable_PRs = []
        self.entry_ids = {e.entry_id for e in self.entries_list}
        self.weight = None
        self.Reactant_record = None
        self.Product_record = {}
        self.min_cost = {}
        self.not_reachable_nodes = []

        self.top_path_list = []
        self.paths = None

        self.solved_PRs = []
        self.PRs_before_final_check = {}

    @classmethod
    def from_input_entries(cls, input_entries, electron_free_energy=-2.15,
                           temperature=298.15, replace_ind = True):
        """
        Generate a ReactionNetwork from a set of MoleculeEntries.

        :param input_entries: list of MoleculeEntries which will make up the
            network
        :param electron_free_energy: float representing the Gibbs free energy
            required to add an electron (in eV)
        :param temperature: Temperature of the system, used for free energy
            and rate constants (in K)
        :param replace_ind: boolean, if True index value of the MoleculeEntry
            will be replace, if false, index value of the MoleculeEntry will
            not be changed
        :return:
        """

        entries = dict()
        entries_list = list()

        print(len(input_entries), "input entries")

        connected_entries = list()
        for entry in input_entries:
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge
        get_free_energy = lambda x: x.free_energy(temp=temperature)

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_Nbonds)
            entries[k1] = dict()
            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                entries[k1][k2] = dict()
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = sorted(list(g3), key=get_free_energy)
                    if len(sorted_entries_3) > 1:
                        unique = list()
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii, Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    if entry.free_energy() is not None and Uentry.free_energy() is not None:
                                        if entry.free_energy(temp=temperature) < Uentry.free_energy(temp=temperature):
                                            unique[ii] = entry
                                    elif entry.free_energy() is not None:
                                        unique[ii] = entry
                                    elif entry.energy < Uentry.energy:
                                        unique[ii] = entry
                                    break
                            if not isomorphic_found:
                                unique.append(entry)
                        entries[k1][k2][k3] = unique
                    else:
                        entries[k1][k2][k3] = sorted_entries_3
                    for entry in entries[k1][k2][k3]:
                        entries_list.append(entry)

        print(len(entries_list), "unique entries")
        if replace_ind:
            for ii, entry in enumerate(entries_list):
                entry.parameters["ind"] = ii

        entries_list = sorted(entries_list,
                              key=lambda x: x.parameters["ind"])

        graph = nx.DiGraph()

        network = cls(electron_free_energy, temperature, entries, entries_list,
                      graph, list(), dict(), dict(), dict(), 0)

        return network

    @staticmethod
    def softplus(free_energy: float) -> float:
        """
            Method to determine edge weight using softplus cost function
        :param free_energy: float
        :return: float
        """
        return float(np.log(1 + (273.0 / 500.0) * np.exp(free_energy)))

    @staticmethod
    def exponent(free_energy: float) -> float:
        """
            Method to determine edge weight using exponent cost function
        :param free_energy: float
        :return: float
        """
        return float(np.exp(free_energy))

    @staticmethod
    def rexp(free_energy: float) -> int:
        """
            Method to determine edge weight using exponent(dG/kt) cost function
        :param free_energy: float
        :return: float
        """

        if free_energy <= 0:
            d = np.array([[free_energy]], dtype=np.float128)
            r = np.exp(d)
        else:
            d = np.array([[free_energy]], dtype=np.float128)
            r = np.exp(38.94*d)
        return r[0][0]

    def build(self, reaction_types=frozenset({"RedoxReaction", "IntramolSingleBondChangeReaction",
                                              "IntermolecularReaction",
                                              "CoordinationBondChangeReaction"})) -> nx.DiGraph:
        """
            A method to build the reaction network graph

        :param reaction_types: set/frozenset of all the reactions class to include while building the graph
        :return: nx.DiGraph
        """

        print("build() start", time.time())
        for entry in self.entries_list:
            self.graph.add_node(entry.parameters["ind"], bipartite=0)

        reaction_types = [load_class(str(self.__module__), s) for s in reaction_types]

        all_reactions = list()
        raw_families = dict()

        for r in reaction_types:
            if r.__name__ == "ConcertedReaction":
                reactions, families = r.generate(self.entries_list)
                all_reactions.append(reactions)
                raw_families[r.__name__] = families
            else:
                reactions, families = r.generate(self.entries)
                all_reactions.append(reactions)
                raw_families[r.__name__] = families

        all_reactions = [i for i in all_reactions if i]
        self.reactions = list(itertools.chain.from_iterable(all_reactions))
        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)

        redox_c = 0
        inter_c = 0
        intra_c = 0
        coord_c = 0

        self.families = dict()
        for label_1, grouping_1 in raw_families.items():
            self.families[label_1] = dict()
            for label_2, grouping_2 in grouping_1.items():
                self.families[label_1][label_2] = dict()
                for label_3 in grouping_2.keys():
                    self.families[label_1][label_2][label_3] = set()

        for ii, r in enumerate(self.reactions):
            r.parameters["ind"] = ii
            if r.reaction_type()["class"] == "RedoxReaction":
                redox_c += 1
                r.electron_free_energy = self.electron_free_energy
            elif r.reaction_type()["class"] == "IntramolSingleBondChangeReaction":
                intra_c += 1
            elif r.reaction_type()["class"] == "IntermolecularReaction":
                inter_c += 1
            elif r.reaction_type()["class"] == "CoordinationBondChangeReaction":
                coord_c += 1
            self.add_reaction(r.graph_representation())

            # TODO: concerted reactions?

            this_class = r.reaction_type()["class"]
            for layer1, class1 in raw_families[this_class].items():
                for layer2, class2 in class1.items():
                    for rxn in class2:
                        # Reactions identical - link by index
                        cond_rct = sorted(r.reactant_ids) == sorted(rxn.reactant_ids)
                        cond_pro = sorted(r.product_ids) == sorted(rxn.product_ids)
                        if cond_rct and cond_pro:
                            self.families[this_class][layer1][layer2].add(ii)

        print("redox: ", redox_c, "inter: ", inter_c, "intra: ", intra_c, "coord: ", coord_c)
        self.PR_record = self.build_PR_record()
        self.Reactant_record = self.build_reactant_record()

        return self.graph

    def add_reaction(self, graph_representation: nx.DiGraph):
        """
            A method to add a single reaction to the ReactionNetwork.graph
            attribute
        :param graph_representation: Graph representation of a reaction,
            obtained from ReactionClass.graph_representation
        """
        self.graph.add_nodes_from(graph_representation.nodes(data=True))
        self.graph.add_edges_from(graph_representation.edges(data=True))

    def build_PR_record(self) -> Mapping_Record_Dict:
        """
            A method to determine all the reaction nodes that have a the same
            PR in the ReactionNetwork.graph

            :return: a dict of the form {int(node1): [all the reaction nodes with
            PR of node1, ex "2+PR_node1, 3"]}
        """
        PR_record = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                PR_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                if "+PR_" in node.split(",")[0]:
                    PR = int(node.split(",")[0].split("+PR_")[1])
                    PR_record[PR].append(node)
        self.PR_record = PR_record
        return PR_record

    def build_reactant_record(self) -> Mapping_Record_Dict:
        """
            A method to determine all the reaction nodes that have the same non
            PR reactant node in the ReactionNetwork.graph

            :return: a dict of the form {int(node1): [all the reaction nodes with
            non PR reactant of node1, ex "node1+PR_2, 3"]}
        """
        Reactant_record = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                Reactant_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                non_PR_reactant = node.split(",")[0].split("+PR_")[0]
                Reactant_record[int(non_PR_reactant)].append(node)
        self.Reactant_record = Reactant_record
        return Reactant_record

    def solve_prerequisites(self, starts: List[int], weight: str, max_iter=25):  # -> Tuple[Union[Dict[Union[int,
        # Any], dict], Any], Any]:
        """
            A method to solve the all the prerequisites found in
            ReactionNetwork.graph. By solving all PRs, it gives information on
            whether 1. if a path exist from any of the starts to all other
            molecule nodes, 2. if so what is the min cost to reach that node
            from any of the start, 3. if there is no path from any of the starts
            to a any of the molecule node, 4. for molecule nodes where the path
            exist, characterize the in the form of ReactionPath
        :param starts: List(molecular nodes), list of molecular nodes of type
            int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use
            when calculating edge weights
        :param max_iter: maximum number of iterations to try to solve all the
            PRs
        :return: PRs: PR_paths: dict that defines a path from each node to a
            start, of the form {int(node1): {int(start1}: {ReactionPath object},
            int(start2): {ReactionPath object}}, int(node2):...}
        :return: old_solved_PRs: list of solved PRs
        """

        print("start solve_prerequisities", time.time())
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        old_attrs = {}
        new_attrs = {}
        self.weight = weight
        self.num_starts = len(starts)
        self.PR_byproducts = {}

        if len(self.graph.nodes) == 0:
            self.build()
        if self.PR_record is None:
            self.PR_record = self.build_PR_record()
        if self.Reactant_record is None:
            self.Reactant_record = self.build_reactant_record()
        orig_graph = copy.deepcopy(self.graph)

        for start in starts:
            PRs[start] = {}

        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = ReactionPath.characterize_path([start], weight, self.min_cost, self.graph)
                else:
                    PRs[PR][start] = ReactionPath(None)

            old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR].cost
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:  # and node != target:
                if node not in PRs:
                    PRs[node] = {}

        ii = 0

        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            print(ii, len(new_solved_PRs) > 0, old_attrs != new_attrs, ii < max_iter)

            min_cost = {}
            cost_from_start = {}
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = float("inf")  # 10000000000000000.0
                self.PR_byproducts[PR] = {}
                for start in PRs[PR]:
                    if PRs[PR][start] == {}:
                        cost_from_start[PR][start] = "no_path"
                    elif PRs[PR][start].path == None:
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start].cost
                        if PRs[PR][start].cost < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start].cost
                            self.PR_byproducts[PR]["byproducts"] = PRs[PR][start].byproducts
                            self.PR_byproducts[PR]["start"] = start
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"
            PRs, cost_from_start, min_cost = self.find_path_cost(starts, weight, old_solved_PRs,
                                                                 cost_from_start, min_cost, PRs)
            solved_PRs = copy.deepcopy(old_solved_PRs)
            solved_PRs, new_solved_PRs, cost_from_start = self.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

            print(ii, len(old_solved_PRs), len(new_solved_PRs), new_solved_PRs)
            attrs = self.update_edge_weights(min_cost, orig_graph)

            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

            print("iteration", ii, "end at", time.time())
            ii += 1
        print("out of while loop at ", time.time())
        self.solved_PRs = copy.deepcopy(old_solved_PRs)
        self.PRs_before_final_check = PRs

        PRs = self.final_PR_check(PRs)
        self.PRs = PRs

        print("total input molecules:", len(self.entries_list), "solvable PRs:", len(old_solved_PRs),
              "unsolvable PRs:", len(self.unsolvable_PRs), 'not reachable mols:', len(self.not_reachable_nodes))
        print("end solve_prerequisities", time.time())
        return PRs, old_solved_PRs

    def parse_path(self, path):
        nodes = []
        PR = []
        Reactants = []
        for step in path:
            if isinstance(step, int):
                nodes.append(step)
            elif "PR_" in step:
                if step.count("+") == 1:
                    nodes = nodes + [step.split("+")[0]]
                    Reactants.append(int(step.split("+")[0]))
                    PR.append(int(step.split("+")[1].split("PR_")[1].split(",")[0]))
                    nodes = nodes + step.split("+")[1].split("PR_")[1].split(",")
                elif step.count("+") == 2:
                    nodes = nodes + [step.split(",")[0].split("+PR_")[0]]
                    Reactants.append(step.split(",")[0].split("+PR_")[0])
                    PR.append(step.split(",")[0].split("+PR_")[1])
                    nodes = nodes + step.split(",")[1].split("+")
                else:
                    print("parse_path something is wrong", path, step)
            else:
                assert (("," in step), True)
                nodes = nodes + step.split(",")
        nodes.pop(0)
        if len(nodes) != 0:
            nodes.pop(-1)
        return nodes, PR, Reactants

    def find_path_cost(self, starts, weight, old_solved_PRs, cost_from_start,
                       min_cost, PRs):
        """
            A method to characterize the path to all the PRs. Characterize by
            determining if the path exist or not, and
            if so, is it a minimum cost path, and if so set PRs[node][start] = ReactionPath(path)
        :param starts: List(molecular nodes), list of molecular nodes of type
            int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the
            ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use
            when calculating edge weights
        :param old_solved_PRs: list of PRs (molecular nodes of type int) that
            are already solved
        :param cost_from_start: dict of type {node1: {start1: float,
                                                      start2: float},
                                              node2: {...}}
        :param min_cost: dict with minimum cost from path start to a node, of
            from {node: float},
            if no path exist, value is "no_path", if path is unsolved yet,
            value is "unsolved_path"
        :param PRs: dict that defines a path from each node to a start,
            of the form {int(node1):
                            {int(start1}: {ReactionPath object},
                            int(start2): {ReactionPath object}},
                         int(node2):...}
        :return: PRs: updated PRs based on new PRs solved
        :return: cost_from_start: updated cost_from_start based on new PRs solved
        :return: min_cost: updated min_cost based on new PRs solved
        """

        not_reachable_nodes_for_start = {}

        wrong_paths = {}
        dist_and_path = {}
        self.num_starts = len(starts)
        for start in starts:
            not_reachable_nodes_for_start[start] = []
            dist, paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(self.graph, start,
                                                                                       weight=self.weight)
            dist_and_path[start] = {}
            wrong_paths[start] = []
            for node in range(len(self.entries_list)):
                if node not in paths.keys():
                    not_reachable_nodes_for_start[start].append(int(node))
            for node in paths:
                if self.graph.nodes[node]["bipartite"] == 0:
                    if node not in self.reachable_nodes:
                        self.reachable_nodes.append(int(node))

                    dist_and_path[start][int(node)] = {}
                    dist_and_path[start][node]["cost"] = dist[node]
                    dist_and_path[start][node]["path"] = paths[node]
                    nodes = []
                    PR = []
                    Reactants = []
                    for step in paths[node]:
                        if isinstance(step, int):
                            nodes.append(step)
                        elif "PR_" in step:
                            if step.count("+") == 1:
                                nodes = nodes + [step.split("+")[0]]
                                Reactants.append(int(step.split("+")[0]))
                                PR.append(int(step.split("+")[1].split("PR_")[1].split(",")[0]))
                                if node in PR:
                                    if node not in wrong_paths[start]:
                                        wrong_paths[start].append(int(node))
                                nodes = nodes + step.split("+")[1].split("PR_")[1].split(",")
                            elif step.count("+") == 2:
                                nodes = nodes + [step.split(",")[0].split("+PR_")[0]]
                                Reactants.append(step.split(",")[0].split("+PR_")[0])
                                PR.append(step.split(",")[0].split("+PR_")[1])
                                if node in PR:
                                    if node not in wrong_paths[start]:
                                        wrong_paths[start].append(int(node))
                                nodes = nodes + step.split(",")[1].split("+")
                            elif step.count("+") == 3:
                                PR.append(step.split(",")[0].split("+PR_")[1])
                                PR.append(step.split(",")[0].split("+PR_")[2])
                                if node in PR:
                                    if node not in wrong_paths[start]:
                                        wrong_paths[start].append(int(node))
                            else:
                                print("SOMETHING IS WRONG", step)
                        else:
                            assert (("," in step), True)
                            nodes = nodes + step.split(",")
                    nodes.pop(0)
                    if len(nodes) != 0:
                        nodes.pop(-1)
                    dist_and_path[start][node]["all_nodes"] = nodes
                    dist_and_path[start][node]["PRs"] = PR
                    dist_and_path[start][node]["reactant"] = Reactants


        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                if node not in self.reachable_nodes:
                    if node not in self.not_reachable_nodes:
                        self.not_reachable_nodes.append(node)


        for start in not_reachable_nodes_for_start:
            for node in not_reachable_nodes_for_start[start]:
                if node not in self.graph.nodes:
                    pass
                else:
                    PRs[node][start] = ReactionPath(None)
                    cost_from_start[node][start] = "no_path"

        fixed_paths = {}
        for start in wrong_paths:
            fixed_paths[start] = {}
            for node in wrong_paths[start]:
                fixed_paths[start][node] = {}
                try:
                    length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                        self.graph,
                        source=hash(start),
                        target=hash(node),
                        ignore_nodes=self.find_or_remove_bad_nodes([node] + self.not_reachable_nodes),
                        weight=self.weight)
                    fixed_paths[start][node]["cost"] = length
                    fixed_paths[start][node]["path"] = dij_path
                except nx.exception.NetworkXNoPath:
                    fixed_paths[start][node]["cost"] = "no_cost"
                    fixed_paths[start][node]["path"] = "no_path"


        self.unsolvable_PRs_per_start = {}
        for start in starts:
            self.unsolvable_PRs_per_start[start] = []
            for node in fixed_paths[start]:
                if fixed_paths[start][node]["path"] == "no_path":
                    dist_and_path[start][node] = {}
                    self.unsolvable_PRs_per_start[start].append(node)
                    pass
                else:
                    dist_and_path[start][node]["cost"] = fixed_paths[start][node]["cost"]
                    dist_and_path[start][node]["path"] = fixed_paths[start][node]["path"]
                    nodes, PR, reactant = self.parse_path(dist_and_path[start][node]["path"])
                    dist_and_path[start][node]["all_nodes"] = nodes
                    dist_and_path[start][node]["PRs"] = PR
                    dist_and_path[start][node]["reactant"] = reactant
            dist_and_path[start] = {key: value for key, value in
                                    sorted(dist_and_path[start].items(), key=lambda item: int(item[0]))}


        for start in starts:
            for node in dist_and_path[start]:
                if node not in old_solved_PRs:
                    if dist_and_path[start][node] == {}:
                        PRs[node][start] = ReactionPath(None)
                        cost_from_start[node][start] = "no_path"
                    elif dist_and_path[start][node]["cost"] == float("inf"):#>= 10000000000000000.0:
                        PRs[node][start] = ReactionPath(None)
                    else:
                        path_class = ReactionPath.characterize_path(dist_and_path[start][node]["path"], weight,
                                                                    self.min_cost, self.graph,
                                                                    old_solved_PRs,
                                                                    PR_byproduct_dict=self.PR_byproducts, actualPRs=PRs)
                        cost_from_start[node][start] = path_class.cost
                        if len(path_class.unsolved_prereqs) == 0:
                            PRs[node][start] = path_class
                        if path_class.cost < min_cost[node]:
                            min_cost[node] = path_class.cost
                            self.PR_byproducts[node]["byproducts"] = path_class.byproducts
                            self.PR_byproducts[node]["start"] = start

        return PRs, cost_from_start, min_cost

    def identify_solved_PRs(self, PRs, solved_PRs, cost_from_start):
        """
            A method to identify new solved PRs after each iteration
        :param PRs: dict that defines a path from each node to a start, of the
            form {int(node1): {int(start1}: {ReactionPath object},
                               int(start2): {ReactionPath object}},
                 int(node2):...}
        :param solved_PRs: list of PRs (molecular nodes of type int) that are
            already solved
        :param cost_from_start: dict of type {node1: {start1: float,
                                                      start2: float},
                                              node2: {...}}
        :return: solved_PRs: list of all the PRs(molecular nodes of type int)
            that are already solved plus new PRs solved in the current iteration
        :return: new_solved_PRs: list of just the new PRs(molecular nodes of
            type int) solved during current iteration
        :return: cost_from_start: updated dict of cost_from_start based on the
            new PRs solved during current iteration
        """
        new_solved_PRs = []

        for PR in PRs:
            if PR not in solved_PRs:
                if len(PRs[PR].keys()) == self.num_starts:
                    new_solved_PRs.append(PR)
                else:
                    best_start_so_far = [None, float("inf")]#10000000000000000.0]
                    for start in PRs[PR]:
                        if PRs[PR][start] is not None:  # ALWAYS TRUE shoudl be != {}
                            if PRs[PR][start].cost < best_start_so_far[1]:
                                best_start_so_far[0] = start
                                best_start_so_far[1] = PRs[PR][start].cost

                    if best_start_so_far[0] is not None:
                        num_beaten = 0
                        for start in cost_from_start[PR]:
                            if start != best_start_so_far[0]:
                                if cost_from_start[PR][start] == "no_path":
                                    num_beaten += 1
                                elif cost_from_start[PR][start] >= best_start_so_far[1]:
                                    num_beaten += 1
                        if num_beaten == self.num_starts - 1:
                            new_solved_PRs.append(PR)

        solved_PRs = solved_PRs + new_solved_PRs


        return solved_PRs, new_solved_PRs, cost_from_start

    def update_edge_weights(self, min_cost: Dict[int, float],
                            orig_graph: nx.DiGraph) -> Dict[Tuple[int, str], Dict[str, float]]:
        """
            A method to update the ReactionNetwork.graph edge weights based on
            the new cost of solving PRs
        :param min_cost: dict with minimum cost from path start to a node, of
            from {node: float}, if no path exist, value is "no_path", if path
            is unsolved yet, value is "unsolved_path"
        :param orig_graph: ReactionNetwork.graph of type nx.Digraph before the
            start of current iteration of updates
        :return: attrs: dict of form {(node1, node2), {"softplus": float,
                                                       "exponent": float,
                                                       "weight: 1},
                                     (node2, node3): {...}}
                dict of all the edges to update the weights of
        """
        if len(self.graph.nodes) == 0:
            self.graph = self.build()
        if self.PR_record is None:
            self.PR_record = self.build_PR_record()

        attrs = {}
        for PR_ind in min_cost:
            for rxn_node in self.PR_record[PR_ind]:
                non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                attrs[(non_PR_reactant_node, rxn_node)] = {
                    self.weight: orig_graph[non_PR_reactant_node][rxn_node][self.weight] + min_cost[PR_ind]}

        nx.set_edge_attributes(self.graph, attrs)
        return attrs

    def final_PR_check(self, PRs: Mapping_PR_Dict):
        """
            A method to check errors in the path attributes of the PRs with a
            path, if no path then prints no path from any start to a given
        :param PRs: dict that defines a path from each node to a start, of the
            form {int(node1): {int(start1}: {ReactionPath object},
                               int(start2): {ReactionPath object}},
                  int(node2):...}
        """
        for PR in PRs:
            path_found = False
            if PRs[PR] != {}:
                for start in PRs[PR]:
                    if PRs[PR][start].cost == float("inf"):  # 10000000000000000.0:
                        PRs[PR][start] = ReactionPath(None)
                    if PRs[PR][start].path != None:
                        path_found = True
                        path_dict_class = ReactionPath.characterize_path_final(PRs[PR][start].path, self.weight,
                                                                               self.min_cost, self.graph,
                                                                               self.solved_PRs,
                                                                               PR_byproduct_dict=self.PR_byproducts,
                                                                               PR_paths=PRs)
                        PRs[PR][start] = path_dict_class
                        if abs(path_dict_class.cost - path_dict_class.pure_cost) > 0.0001:
                            print("WARNING: cost mismatch for PR", PR, path_dict_class.cost, path_dict_class.pure_cost,
                                  path_dict_class.path_dict, path_dict_class.full_path)

                if not path_found:
                    print("No path found from any start to PR", PR)
            else:
                self.unsolvable_PRs.append(PR)
                print("Unsolvable path from any start to PR", PR)
        self.PRs = PRs
        return PRs

    def remove_node(self, node_ind):
        '''
        Remove a species from self.graph. Also remove all the reaction nodes with that species. Used for removing Li0.
        :param: list of node numbers to remove
        '''
        for n in node_ind:
            self.graph.remove_node(n)
            nodes = list(self.graph.nodes)
            for node in nodes:
                if self.graph.nodes[node]["bipartite"] == 1:
                    reactants = node.split(',')[0].split('+')
                    reactants = [reac.replace('PR_', '') for reac in reactants]
                    products = node.split(',')[1].split('+')
                    if str(n) in products:
                        if len(reactants) == 2:
                            self.PR_record[int(reactants[1])].remove(node)
                            self.graph.remove_node(node)
                            self.PR_record.pop(node, None)
                    elif str(n) in reactants:
                        if len(reactants) == 2:
                            self.PR_record[int(reactants[1])].remove(node)
                        self.Reactant_record.pop(node, None)

                        self.graph.remove_node(node)
            self.PR_record.pop(n, None)
            self.Product_record.pop(n,None)


    def find_or_remove_bad_nodes(self, nodes: List[str],
                                 remove_nodes=False) -> List[str] or nx.DiGraph:
        """
            A method to either create a list of the nodes a path solving method
            should ignore or generate a graph without all the nodes it a path
            solving method should not use in obtaining a path.
        :param nodes: List(molecular nodes), list of molecular nodes of type int
            found in the ReactionNetwork.graph
            that should be ignored when solving a path
        :param remove_nodes: if False (default), return list of bad nodes, if
            True, return a version of ReactionNetwork.graph (of type nx.Digraph)
            from with list of bad nodes are removed
        :return: if remove_nodes = False -> list[node],
                 if remove_nodes = True -> nx.DiGraph
        """
        if len(self.graph.nodes) == 0:
            self.graph = self.build()
        if self.PR_record is None:
            self.PR_record = self.build_PR_record()
        if self.Reactant_record is None:
            self.Reactant_record = self.build_reactant_record()
        bad_nodes = []
        for node in nodes:
            for bad_node in self.PR_record[node]:
                bad_nodes.append(bad_node)
            for bad_nodes2 in self.Reactant_record[node]:
                bad_nodes.append(bad_nodes2)
        if remove_nodes:
            pruned_graph = copy.deepcopy(self.graph)
            pruned_graph.remove_nodes_from(bad_nodes)
            return pruned_graph
        else:
            return bad_nodes

    def valid_shortest_simple_paths(self, start: int, target: int, PRs=[]):  # -> Generator[List[str]]:????
        """
            A method to determine shortest path from start to target
        :param start: molecular node of type int from ReactionNetwork.graph
        :param target: molecular node of type int from ReactionNetwork.graph
        :param PRs: not used currently?
        :return: nx.path_generator of type generator
        """
        valid_graph = self.find_or_remove_bad_nodes([target], remove_nodes=True)
        valid_graph.remove_nodes_from(PRs)

        return nx.shortest_simple_paths(valid_graph, hash(start), hash(target), weight=self.weight)


    def find_paths(self, starts, target, weight, num_paths=10, ignorenode=[]):  # -> ??
        """
            A method to find the shorted parth from given starts to a target

        :param starts: starts: List(molecular nodes), list of molecular nodes
            of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the
            ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use
            when calculating edge weights
        :param num_paths: Number (of type int) of paths to find. Defaults to 10.
        :param solved_PRs_path: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object},
                                          int(start2): {ReactionPath object}},
                                          int(node2):...}
                if None, method will solve PRs
        :param solved_min_cost: dict with minimum cost from path start to a
                node, of from {node: float}, if no path exist, value is
                "no_path", if path is unsolved yet, value is "unsolved_path",
                of None, method will solve for min_cost
        :param updated_graph: nx.DiGraph with udpated edge weights based on
            the solved PRs, if none, method will solve for PRs and update graph
            accordingly
        :param save: if True method will save PRs paths, min cost and updated
                    graph after all the PRs are solved,
                    if False, method will not save anything (default)
        :return: PR_paths: solved dict of PRs
        :return: paths: list of paths (number of paths based on the value of
            num_paths)
        """

        print("find_paths start", time.time())
        self.weight = weight
        self.num_starts = len(starts)
        paths = []
        c = itertools.count()
        my_heapq = []
        if self.PRs == {}:
            print("Solving prerequisites...")
            if len(self.graph.nodes) == 0:
                self.build()
            self.solve_prerequisites(starts, weight)

        print("Finding paths...")

        remove_node = []
        for PR in self.unsolvable_PRs:
            remove_node = remove_node + self.PR_record[PR]
        ignorenode = ignorenode + remove_node
        try:
            for start in starts:
                ind = 0
                print(start, target)
                for path in self.valid_shortest_simple_paths(start, target, ignorenode):
                    # print(ind, path)
                    if ind == num_paths:
                        break
                    else:
                        ind += 1
                        path_dict_class2 = ReactionPath.characterize_path_final(path, self.weight, self.min_cost,
                                                                                self.graph, self.solved_PRs,
                                                                                PR_byproduct_dict=self.PR_byproducts,
                                                                                PR_paths=self.PRs)
                        heapq.heappush(my_heapq, (path_dict_class2.cost, next(c), path_dict_class2))
        except:
            print("ind", ind)
        top_path_list = []
        while len(paths) < num_paths and my_heapq:
            (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
            top_path_list.append(path_dict_HP_class.path)
            print(len(paths), cost_HP, path_dict_HP_class.overall_free_energy_change,
                  path_dict_HP_class.hardest_step_deltaG, path_dict_HP_class.path_dict)
            paths.append(
                path_dict_HP_class.path_dict)  ### ideally just append the class, but for now dict for easy printing

        self.paths = paths
        self.top_path_list = top_path_list
        print("find_paths end", time.time())

        return self.PRs, paths, top_path_list

    @staticmethod
    def mols_w_cuttoff(RN_pr_solved, cutoff=0, build_pruned_network=True):
        """"
            A method to identify molecules reached by dG <= cutoff
        :param RN_pr_solved: instance of reaction network
        :param: cutoff: dG value
        :return: mols_to_keep: list of molecule nodes that can be reached by dG <= cutoff
        :return: pruned_entries_list: list of MoleculeEntry of molecules that can be reached by dG <= cutoff
        """

        pruned_PRs = {}
        for PR_node in RN_pr_solved.PR_byproducts:
            if RN_pr_solved.PRs[PR_node] != {} and RN_pr_solved.PR_byproducts[PR_node] != {}:
                min_start = RN_pr_solved.PR_byproducts[PR_node]["start"]
                if RN_pr_solved.PRs[PR_node][min_start].overall_free_energy_change <= cutoff:
                    pruned_PRs[PR_node] = {}
                    pruned_PRs[PR_node][min_start] = RN_pr_solved.PRs[PR_node][min_start]

        nodes_to_keep = []
        for PR_node in pruned_PRs:
            for start in pruned_PRs[PR_node]:
                nodes_to_keep = nodes_to_keep + pruned_PRs[PR_node][start].full_path

        nodes_to_keep = list(dict.fromkeys(nodes_to_keep))
        mols_to_keep = []
        for node in nodes_to_keep:
            if isinstance(node, int):
                mols_to_keep.append(node)
        mols_to_keep.sort()

        pruned_entries_list = []
        for entry in RN_pr_solved.entries_list:
            if entry.parameters["ind"] in mols_to_keep:
                pruned_entries_list.append(entry)

        if build_pruned_network:
            pruned_network_build = ReactionNetwork.from_input_entries(pruned_entries_list, replace_ind=False)
            pruned_network_build.build()
            return mols_to_keep, pruned_entries_list, pruned_network_build
        else:
            return mols_to_keep, pruned_entries_list


    @staticmethod
    def identify_concerted_rxns_via_intermediates(RN_pr_solved, mols_to_keep,
                                                  single_elem_interm_ignore=["C1", "H1", "O1", "Li1"]):
        """
            A method to identify concerted reactions by looping through high enery intermediates
        :param RN_pr_solved: ReactionNetwork that is PR solved
        :param mols_to_keep: List of pruned molecules, if not running then a list of all molecule nodes in the
        RN_pr_solved
        :param single_elem_interm_ignore: List of formula of high energy intermediates to ignore
        :return: list of reactions
        """

        flag = True
        print("identify_concerted_rxns_via_intermediates start", time.time())
        mols_to_keep.append(None)
        count_total = 0
        reactions = []
        not_wanted_formula = single_elem_interm_ignore
        for entry in RN_pr_solved.entries_list:
            node = entry.parameters["ind"]
            if RN_pr_solved.entries_list[node].formula not in not_wanted_formula and RN_pr_solved.graph.nodes[node][
                "bipartite"] == 0 and node not in RN_pr_solved.not_reachable_nodes and node not in RN_pr_solved.unsolvable_PRs:
                out_nodes = []
                for rxn in list(RN_pr_solved.graph.neighbors(node)):
                    if "electron" not in RN_pr_solved.graph.nodes[rxn]["rxn_type"]:
                        out_nodes.append(rxn)
                in_nodes = []
                for in_edge in list(RN_pr_solved.graph.in_edges(node)):
                    in_rxn = in_edge[0]
                    if "electron" not in RN_pr_solved.graph.nodes[in_rxn]["rxn_type"]:
                        in_nodes.append(in_rxn)
                count = 0
                for out_node in out_nodes:
                    for in_node in in_nodes:
                        rxn1_dG = RN_pr_solved.graph.nodes[in_node]["free_energy"]
                        total_dG = rxn1_dG + RN_pr_solved.graph.nodes[out_node]["free_energy"]
                        if rxn1_dG > 0 and total_dG < 0:
                            # if flag:
                            if "PR" in out_node and "PR" in in_node:
                                pass
                            elif "PR" not in out_node and "PR" not in in_node:
                                if "+" in out_node and "+" in in_node:
                                    pass
                                elif "+" not in out_node and "+" not in in_node:
                                    if in_node.split(",")[0] == out_node.split(",")[1]:
                                        pass
                                    else:
                                        in_mol = in_node.split(",")[0]
                                        out_mol = out_node.split(",")[1]
                                        glist = [int(in_mol), int(out_mol)]
                                        gnode = in_mol + "," + out_mol
                                        reactant = int(in_mol)
                                        product = int(out_mol)
                                        glist = [reactant, product]
                                        if set(glist).issubset(set(mols_to_keep)):
                                            count = count + 1
                                            reactions.append(([reactant], [product], [in_node, out_node]))
                                            # print(([reactant], [product]), in_node, out_node)
                                elif "+" in in_node and "+" not in out_node:
                                    reactant = int(in_node.split(",")[0])
                                    product1 = in_node.split(",")[1].split("+")
                                    product1.remove(str(node))
                                    product1 = int(product1[0])
                                    product2 = int(out_node.split(",")[1])
                                    glist = [int(reactant), int(product1), int(product2)]
                                    if set(glist).issubset(set(mols_to_keep)):
                                        count = count + 1
                                        reactions.append(([reactant], [product1, product2], [in_node, out_node]))
                                elif "+" not in in_node and "+" in out_node:
                                    reactant = int(in_node.split(",")[0])
                                    product1 = int(out_node.split(",")[1].split("+")[0])
                                    product2 = int(out_node.split(",")[1].split("+")[1])
                                    glist = [int(reactant), int(product1), int(product2)]
                                    if set(glist).issubset(set(mols_to_keep)):
                                        count = count + 1
                                        reactions.append(([reactant], [product1, product2], [in_node, out_node]))
                            elif "PR" in in_node and "PR" not in out_node:
                                if "+" in out_node:
                                    p1 = list(map(int, out_node.split(",")[1].split("+")))[0]
                                    p2 = list(map(int, out_node.split(",")[1].split("+")))[1]
                                else:
                                    p2 = out_node.split(",")[1]
                                    p1 = None
                                PR = in_node.split(",")[0].split("+PR_")[1]
                                start = in_node.split("+")[0]
                                if p1 is None:
                                    reactant1 = int(start)
                                    reactant2 = int(PR)
                                    product1 = int(p2)
                                    product2 = None
                                    glist = [reactant1, reactant2, product1]
                                else:
                                    reactant1 = int(start)
                                    reactant2 = int(PR)
                                    product1 = int(p2)
                                    product2 = int(p1)
                                    glist = [reactant1, reactant2, product1, product2]
                                if set(glist).issubset(set(mols_to_keep)) and {reactant1, reactant2} != {product1,
                                                                                                         product2}:
                                    count = count + 1
                                    # print(glist, set(glist).issubset(set(mols_to_keep)))
                                    reactions.append(
                                        ([reactant1, reactant2], [product1, product2], [in_node, out_node]))
                            elif "PR" not in in_node and "PR" in out_node:
                                if "+" in in_node:  #####want this: 2441,2426''2426+PR_148,3669'
                                    start = in_node.split(",")[0]
                                    p1 = list(map(int, in_node.split(",")[1].split("+")))
                                    p1.remove(node)
                                    p1 = p1[0]
                                else:
                                    start = in_node.split(",")[0]
                                    p1 = None
                                PR = out_node.split(",")[0].split("+PR_")[1]
                                p2 = out_node.split(",")[1]
                                if p1 is None:
                                    reactant1 = int(start)
                                    reactant2 = int(PR)
                                    product1 = int(p2)
                                    product2 = None
                                    glist = [reactant1, reactant2, product1, product2]

                                else:
                                    reactant1 = int(start)
                                    reactant2 = int(PR)
                                    product1 = int(p1)
                                    product2 = int(p2)
                                    glist = [reactant1, reactant2, product1, product2]

                                if set(glist).issubset(set(mols_to_keep)) and {reactant1, reactant2} != {product1,
                                                                                                         product2}:
                                    count = count + 1
                                    # print(glist, set(glist).issubset(set(mols_to_keep)))
                                    reactions.append(
                                        ([reactant1, reactant2], [product1, product2], [in_node, out_node]))
                count_total = count_total + count
                print(node, entry.formula, count)
        print("identify_concerted_rxns_via_intermediates end", time.time())
        print("total number of unique concerted reactions:", count_total)

        return reactions

    @staticmethod
    def add_concerted_rxns(full_network_pr_solved, pruned_network_build, reactions):
        """
            A method to add concerted reactions (obtained from identify_concerted_rxns_via_intermediates() method)to
            the ReactonNetwork
        :param full_network_pr_solved: full network that is not pruned
        :param pruned_network_build: network that is pruned, if not pruning, use the same network
        :param reactions: list of reactions obtained from identify_concerted_rxns_via_intermediates() method
        :return: pruned network with concerted reactions added
        """

        print("add_concerted_rxns start", time.time())
        c1 = 0
        c2 = 0
        c3 = 0
        for reaction in reactions:
            if len(reaction[0]) == 1 and len(reaction[1]) == 1:
                # print(reaction)
                assert int(reaction[0][0]) in pruned_network_build.graph.nodes
                assert int(reaction[1][0]) in pruned_network_build.graph.nodes
                reactants = full_network_pr_solved.entries_list[int(reaction[0][0])]
                products = full_network_pr_solved.entries_list[int(reaction[1][0])]
                cr = ConcertedReaction([reactants], [products])
                cr.electron_free_energy = -2.15
                g = cr.graph_representation()
                for node in list(g.nodes):
                    if not isinstance(node, int) and g.nodes[node]["free_energy"] > 0:
                        g.remove_node(node)
                pruned_network_build.add_reaction(g)
                c1 = c1 + 1
            elif len(reaction[0]) == 1 and len(reaction[1]) == 2:
                # print(reaction)
                assert int(reaction[0][0]) in pruned_network_build.graph.nodes
                assert int(reaction[1][0]) in pruned_network_build.graph.nodes
                assert int(reaction[1][1]) in pruned_network_build.graph.nodes
                reactant_0 = full_network_pr_solved.entries_list[int(reaction[0][0])]
                product_0 = full_network_pr_solved.entries_list[int(reaction[1][0])]
                product_1 = full_network_pr_solved.entries_list[int(reaction[1][1])]
                cr = ConcertedReaction([reactant_0], [product_0, product_1])
                cr.electron_free_energy = -2.15
                g = cr.graph_representation()
                for node in list(g.nodes):
                    if not isinstance(node, int) and g.nodes[node]["free_energy"] > 0:
                        g.remove_node(node)
                pruned_network_build.add_reaction(g)
                c2 = c2 + 1
            elif len(reaction[0]) == 2 and len(reaction[1]) == 2:
                assert int(reaction[0][0]) in pruned_network_build.graph.nodes
                assert int(reaction[0][1]) in pruned_network_build.graph.nodes
                assert int(reaction[1][0]) in pruned_network_build.graph.nodes
                new_node = False
                if reaction[1][1] == None:
                    new_node = True
                else:
                    assert int(reaction[1][1]) in pruned_network_build.graph.nodes
                if not new_node:
                    reactant_0 = full_network_pr_solved.entries_list[int(reaction[0][0])]
                    PR = full_network_pr_solved.entries_list[int(reaction[0][1])]
                    product_0 = full_network_pr_solved.entries_list[int(reaction[1][0])]
                    product_1 = full_network_pr_solved.entries_list[int(reaction[1][1])]
                    cr = ConcertedReaction([reactant_0, PR], [product_0, product_1])
                    cr.electron_free_energy = -2.15
                    g = cr.graph_representation()
                    for node in list(g.nodes):
                        if not isinstance(node, int) and g.nodes[node]["free_energy"] > 0:
                            g.remove_node(node)
                    pruned_network_build.add_reaction(g)
                    c3 = c3 + 1
                elif new_node:
                    reactant_0 = full_network_pr_solved.entries_list[int(reaction[0][0])]
                    PR = full_network_pr_solved.entries_list[int(reaction[0][1])]
                    product_0 = full_network_pr_solved.entries_list[int(reaction[1][0])]
                    cr = ConcertedReaction([reactant_0, PR], [product_0])
                    cr.electron_free_energy = -2.15
                    g = cr.graph_representation()
                    for node in list(g.nodes):
                        if not isinstance(node, int) and g.nodes[node]["free_energy"] > 0:
                            g.remove_node(node)
                    pruned_network_build.add_reaction(g)
                    c3 = c3 + 1

        pruned_network_build.PR_record = pruned_network_build.build_PR_record()
        pruned_network_build.Reactant_record = pruned_network_build.build_reactant_record()
        print("add_concerted_rxns end", time.time())
        return pruned_network_build

    def as_dict(self) -> dict:
        entries = dict()
        for formula in self.entries.keys():
            entries[formula] = dict()
            for bonds in self.entries[formula].keys():
                entries[formula][bonds] = dict()
                for charge in self.entries[formula][bonds].keys():
                    entries[formula][bonds][charge] = list()
                    for entry in self.entries[formula][bonds][charge]:
                        entries[formula][bonds][charge].append(entry.as_dict())

        entries_list = [e.as_dict() for e in self.entries_list]

        reactions = [r.as_dict() for r in self.reactions]

        families = dict()
        for category in self.families.keys():
            families[category] = dict()
            for charge in self.families[category].keys():
                families[category][charge] = dict()
                for label in self.families[category][charge].keys():
                    families[category][charge][label] = list()
                    for reaction in self.families[category][charge][label]:
                        families[category][charge][label].append(reaction)

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "entries_dict": entries,
             "entries_list": entries_list,
             "reactions": reactions,
             "families": families,
             "electron_free_energy": self.electron_free_energy,
             "temperature": self.temperature,
             "graph": json_graph.adjacency_data(self.graph),
             "PR_record": self.PR_record,
             "min_cost": self.min_cost,
             "num_starts": self.num_starts}

        return d


    @classmethod
    def from_dict(cls, d):

        entries = dict()
        d_entries = d["entries_dict"]
        for formula in d_entries.keys():
            entries[formula] = dict()
            for bonds in d_entries[formula].keys():
                int_bonds = int(bonds)
                entries[formula][int_bonds] = dict()
                for charge in d_entries[formula][bonds].keys():
                    int_charge = int(charge)
                    entries[formula][int_bonds][int_charge] = list()
                    for entry in d_entries[formula][bonds][charge]:
                        entries[formula][int_bonds][int_charge].append(MoleculeEntry.from_dict(entry))

        entries_list = [MoleculeEntry.from_dict(e) for e in d["entries_list"]]

        reactions = list()
        for reaction in d["reactions"]:
            rclass = load_class(str(cls.__module__), reaction["@class"])
            reactions.append(rclass.from_dict(reaction))

        families = dict()
        for category in d["families"].keys():
            families[category] = dict()
            for layer_one in d["families"][category].keys():
                families[category][layer_one] = dict()
                for layer_two in d["families"][category][layer_one].keys():
                    families[category][layer_one][layer_two] = list()
                    for reaction in d["families"][category][layer_one][layer_two]:
                        families[category][layer_one][layer_two].append(reaction)

        graph = json_graph.adjacency_graph(d["graph"], directed=True)

        return cls(d["electron_free_energy"], d["temperature"], entries,
                   entries_list, graph, reactions, families,
                   d["PR_record"], d["min_cost"], d["num_starts"])



def graph_rep_3_2(reaction: Reaction) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 3 reactants -> 2
    products
    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction, Concerted)
    """

    if len(reaction.reactants) != 3 or len(reaction.products) != 2:
        raise ValueError("Must provide reaction with 3 reactants and 2 products for graph_rep_3_2")

    reactant_0 = reaction.reactants[0]
    reactant_1 = reaction.reactants[1]
    reactant_2 = reaction.reactants[2]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_prod_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_prod_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_prod_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_prod_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    reactants_ind_list = np.array(
        [reactant_0.parameters["ind"], reactant_1.parameters["ind"], reactant_2.parameters["ind"]])
    reactant_inds = np.argsort(reactants_ind_list)
    reactants_ind_list = np.sort(reactants_ind_list)

    reactants_name = str(reactants_ind_list[0]) + "+" + str(reactants_ind_list[1]) + "+" + str(reactants_ind_list[2])
    reactants_name_entry_ids = str(reactants_ind_list[reactant_inds[0]]) + "+" + str(
        reactants_ind_list[reactant_inds[1]]) + "+" + str(reactants_ind_list[reactant_inds[2]])

    two_prod_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_prod_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])

    if reactant_1.parameters["ind"] <= reactant_2.parameters["ind"]:
        three_reac_name0 = str(reactant_0.parameters["ind"]) + "+PR_" + str(
            reactant_1.parameters["ind"]) + "+PR_" + str(reactant_2.parameters["ind"])
        three_reac_entry_ids0 = str(reactant_0.entry_id) + "+PR_" + str(reactant_1.entry_id) + "+PR_" + str(
            reactant_2.entry_id)
    else:
        three_reac_name0 = str(reactant_0.parameters["ind"]) + "+PR_" + str(
            reactant_2.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
        three_reac_entry_ids0 = str(reactant_0.entry_id) + "+PR_" + str(reactant_2.entry_id) + "+PR_" + str(
            reactant_1.entry_id)
    if reactant_0.parameters["ind"] <= reactant_2.parameters["ind"]:
        three_reac_name1 = str(reactant_1.parameters["ind"]) + "+PR_" + str(
            reactant_0.parameters["ind"]) + "+PR_" + str(reactant_2.parameters["ind"])
        three_reac_entry_ids1 = str(reactant_1.entry_id) + "+PR_" + str(reactant_0.entry_id) + "+PR_" + str(
            reactant_2.entry_id)
    else:
        three_reac_name1 = str(reactant_1.parameters["ind"]) + "+PR_" + str(
            reactant_2.parameters["ind"]) + "+PR_" + str(reactant_0.parameters["ind"])
        three_reac_entry_ids1 = str(reactant_1.entry_id) + "+PR_" + str(reactant_2.entry_id) + "+PR_" + str(
            reactant_0.entry_id)
    if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
        three_reac_name2 = str(reactant_2.parameters["ind"]) + "+PR_" + str(
            reactant_0.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
        three_reac_entry_ids2 = str(reactant_2.entry_id) + "+PR_" + str(reactant_0.entry_id) + "+PR_" + str(
            reactant_1.entry_id)
    else:
        three_reac_name2 = str(reactant_2.parameters["ind"]) + "+PR_" + str(
            reactant_1.parameters["ind"]) + "+PR_" + str(reactant_0.parameters["ind"])
        three_reac_entry_ids2 = str(reactant_2.entry_id) + "+PR_" + str(reactant_1.entry_id) + "+PR_" + str(
            reactant_0.entry_id)

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

    graph.add_node(node_name_A0, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A0)

    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A0,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A0,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A0,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_A1, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A1)

    graph.add_edge(reactant_1.parameters["ind"],
                   node_name_A1,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A1,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A1,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_A2, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A2)

    graph.add_edge(reactant_2.parameters["ind"],
                   node_name_A2,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A1,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A1,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)

    graph.add_edge(product_0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_B0,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B0,
                   reactant_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B0,
                   reactant_2.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(product_1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_B1,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_2.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
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
        raise ValueError("Must provide reaction with 2 reactants and 2 products for graph_rep_2_2")

    reactant_0 = reaction.reactants[0]
    reactant_1 = reaction.reactants[1]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_prod_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_prod_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_prod_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_prod_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
        two_reac_name = str(reactant_0.parameters["ind"]) + "+" + str(reactant_1.parameters["ind"])
        two_reac_name_entry_ids = str(reactant_0.entry_id) + "+" + str(reactant_1.entry_id)
    else:
        two_reac_name = str(reactant_1.parameters["ind"]) + "+" + str(reactant_0.parameters["ind"])
        two_reac_name_entry_ids = str(reactant_1.entry_id) + "+" + str(reactant_0.entry_id)

    two_prod_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_prod_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])

    two_reac_name0 = str(reactant_0.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
    two_reac_name1 = str(reactant_1.parameters["ind"]) + "+PR_" + str(reactant_0.parameters["ind"])

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

    graph.add_node(node_name_A0, rxn_type=rxn_type_A, bipartite=1,
                   energy=energy_A,
                   free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A0)

    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A0,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A0,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A0,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_A1, rxn_type=rxn_type_A, bipartite=1,
                   energy=energy_A,
                   free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A1)

    graph.add_edge(reactant_1.parameters["ind"],
                   node_name_A1,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A1,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A1,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1,
                   energy=energy_B,
                   free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)

    graph.add_edge(product_0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_B),
                   weight=1.0
                   )

    graph.add_edge(node_name_B0,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B0,
                   reactant_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1,
                   energy=energy_B,
                   free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(product_1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_B),
                   weight=1.0
                   )

    graph.add_edge(node_name_B1,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
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
        raise ValueError("Must provide reaction with 1 reactant and 2 products"
                         "for graph_rep_1_2")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_mol_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_mol_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_mol_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_mol_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    two_mol_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_mol_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
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

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1,
                   energy=energy_A,
                   free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A)

    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1,
                   energy=energy_B,
                   free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)
    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1,
                   energy=energy_B,
                   free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(node_name_B0,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0
                   )

    graph.add_edge(product_0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_B),
                   weight=1.0
                   )
    graph.add_edge(product_1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_B),
                   weight=1.0)
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
        raise ValueError("Must provide reaction with 1 reactant and product"
                         "for graph_rep_1_1")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    graph = nx.DiGraph()
    node_name_A = str(reactant_0.parameters["ind"]) + "," + str(product_0.parameters["ind"])
    node_name_B = str(product_0.parameters["ind"]) + "," + str(reactant_0.parameters["ind"])
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(reactant_0.entry_id) + "," + str(product_0.entry_id)
    entry_ids_B = str(product_0.entry_id) + "," + str(reactant_0.entry_id)

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1,
                   energy=energy_A,
                   free_energy=free_energy_A,
                   entry_ids=entry_ids_A)
    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   rexp=ReactionNetwork.rexp(free_energy_A),
                   weight=1.0)
    graph.add_edge(node_name_A,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0)
    graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1,
                   energy=energy_B,
                   free_energy=free_energy_B,
                   entry_ids=entry_ids_B)
    graph.add_edge(product_0.parameters["ind"],
                   node_name_B,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   rexp=ReactionNetwork.rexp(free_energy_B),
                   weight=1.0)
    graph.add_edge(node_name_B,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   rexp=0.0,
                   weight=1.0)
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