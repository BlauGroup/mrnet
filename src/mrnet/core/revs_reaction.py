from abc import ABCMeta, abstractmethod
import copy
import itertools
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.constants import h, k, R

import networkx as nx
import networkx.algorithms.isomorphism as iso

from monty.json import MSONable
from monty.serialization import loadfn

from pymatgen.analysis.graphs import MolGraphSplitError
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.reaction_network.reaction_rates import (
    ReactionRateCalculator,
    ExpandedBEPRateCalculator,
    RedoxRateCalculator,
)


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

    Assumes that the reactants and products will never change post-instantiation
    Args:
         reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
         products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2.
         transition_state (MoleculeEntry or None): A MoleculeEntry representing a
             transition state for the reaction.
         parameters (dict): Any additional data about this reaction
    """

    def __init__(
        self,
        reactants: List[MoleculeEntry],
        products: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
    ):
        self.transition_state = transition_state
        if self.transition_state is None:
            # Provide no reference initially
            self.rate_calculator = None
        else:
            self.rate_calculator = ReactionRateCalculator(
                reactants, products, self.transition_state
            )

        self.reactant_ids = [r.entry_id for r in reactants]
        self.product_ids = [p.entry_id for p in products]
        self.r_dicts = [r.as_dict() for r in reactants]
        self.p_dicts = [p.as_dict() for p in products]
        self.entry_ids = {e.entry_id for e in reactants}
        self.parameters = parameters or dict()
        self.rct_indices = [r.parameters.get("ind") for r in reactants]
        self.pro_indices = [p.parameters.get("ind") for p in products]

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactant_ids)

    def update_calculator(
        self, transition_state: Optional[MoleculeEntry] = None, reference: Optional[Dict] = None,
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
                # Critical assumption-neither reactants nor products are being updated.
                self.rate_calculator.update_calc(reference)
        else:
            self.rate_calculator.transition_state = transition_state

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
            "reactants": self.r_dicts,
            "products": self.p_dicts,
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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        reaction = cls(reactants, products, transition_state=ts, parameters=d["parameters"])
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
    ):
        """
          Initilizes RedoxReaction.reactant to be in the form of a MoleculeEntry,
          RedoxReaction.product to be in the form of MoleculeEntry,
          Reaction.reactant to be in the form of a of a list of MoleculeEntry of length 1
          Reaction.products to be in the form of a of a list of MoleculeEntry of length 1

        Args:
          reactant: MoleculeEntry object
          product: MoleculeEntry object
          inner_reorganization_energy (float): Inner reorganization energy, in eV
          dielectric (float): Dielectric constant of the solvent
          refractive (float): Refractive index of the solvent
          electron_free_energy (float): Free energy of the electron in the
              electrode, in eV
          radius (float): Solute cavity radius (including inner solvent shell)
          electrode_dist (float): Distance from reactants to electrode, in
              Angstrom
          parameters (dict): Any additional data about this reaction

        """

        super().__init__(
            [reactant], [product], transition_state=None, parameters=parameters,
        )
        self.class_type = "RedoxReaction"
        self.inner_reorganization_energy = inner_reorganization_energy
        self.dielectric = dielectric
        self.refractive = refractive
        self.electron_free_energy = electron_free_energy
        self.radius = radius
        self.electrode_dist = electrode_dist

        if all(
            [
                self.inner_reorganization_energy,
                self.dielectric,
                self.refractive,
                self.electron_free_energy,
                self.radius,
                self.electrode_dist,
            ]
        ):
            self.rate_calculator = RedoxRateCalculator(
                [reactant],
                [product],
                self.inner_reorganization_energy,
                self.dielectric,
                self.refractive,
                self.electron_free_energy,
                self.radius,
                self.electrode_dist,
            )
        self.reaction_type(reactant, product)
        self.e0_free_energy_method = reactant.free_energy
        self.e1_free_energy_method = product.free_energy
        self.rct_energy = reactant.energy
        self.pro_energy = product.energy
        self.num_rct = 1
        self.num_pro = 1

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a RedoxReaction class object into graph representation (nx.Digraph object).
        Redox Reaction must be of type 1 reactant -> 1 product

        :return nx.Digraph object of a single Redox Reaction
        """

        return graph_rep_1_1(self)

    def update_calculator(
        self, transition_state: Optional[MoleculeEntry] = None, reference: Optional[Dict] = None,
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
        Returns:
            None
        """

        if reference is None:
            pass
        else:
            self.rate_calculator.update_calc(reference)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction], Mapping_Family_Dict]:
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
                                [
                                    reactions.append(cls(entry0, entry1))
                                    for entry1 in entries[formula][Nbonds][charge1]
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph)
                                ]
                                [
                                    families[formula][charge0].append(cls(entry0, entry1))
                                    for entry1 in entries[formula][Nbonds][charge1]
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph)
                                ]

        return reactions, families

    def reaction_type(self, reactant, product):
        """
        A method to identify type of redox reaction (oxidation or reduction)

        Args:
           :return dictionary of the form {"class": "RedoxReaction",
                "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
           where rnx_type_A is the primary type of the reaction based on the
                reactant and product of the RedoxReaction
           object, and the backwards of this reaction would be rnx_type_B
        """
        types = ["One electron oxidation", "One electron reduction"]
        self.rxn_type_A = types[int(product.charge < reactant.charge)]
        self.rxn_type_B = types[1 - int(product.charge < reactant.charge)]

    def free_energy(self, temperature=298.15):
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

        if self.e1_free_energy_method is not None and self.e0_free_energy_method is not None:
            free_energy_A = self.e1_free_energy_method(
                temp=temperature
            ) - self.e0_free_energy_method(temp=temperature)
            free_energy_B = self.e0_free_energy_method(
                temp=temperature
            ) - self.e1_free_energy_method(temp=temperature)

            if self.rxn_type_A == "One electron reduction":
                free_energy_A -= self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B -= self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None
        self.free_energy_A = free_energy_A
        self.free_energy_B = free_energy_B

    def energy(self):
        """
        A method to determine the energy of the redox reaction

        Args:
           :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
           where energy_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
           object, and the backwards of this reaction would be energy_B.
        """
        if self.pro_energy is not None and self.rct_energy is not None:
            self.energy_A = self.pro_energy - self.rct_energy
            self.energy_B = 0 - self.energy_A
        else:
            self.energy_A = None
            self.energy_B = None

    def rate_constant(self, temperature=298.15):
        if isinstance(self.rate_calculator, RedoxRateCalculator):
            self.k_A = self.rate_calculator.calculate_rate_constant(temperature=temperature)
            self.k_B = self.rate_calculator.calculate_rate_constant(
                temperature=temperature, reverse=True
            )
        else:

            if self.electrode_dist is None:
                kappa = 1
            else:
                kappa = np.exp(-1.2 * self.electrode_dist)

            if self.inner_reorganization_energy is None:
                delta_g_a = self.free_energy_A
                delta_g_b = self.free_energy_B
            else:
                lam_reorg = self.inner_reorganization_energy
                delta_g_a = lam_reorg / 4 * (1 + self.free_energy_A / lam_reorg) ** 2
                delta_g_b = lam_reorg / 4 * (1 + self.free_energy_B / lam_reorg) ** 2

            if self.inner_reorganization_energy is None and self.free_energy_A < 0:
                self.k_A = kappa * k * temperature / h
            else:
                self.k_A = (
                    kappa * k * temperature / h * np.exp(-96487 * delta_g_a / (R * temperature))
                )

            if self.inner_reorganization_energy is None and self.free_energy_B < 0:
                self.k_B = kappa * k * temperature / h
            else:
                self.k_B = (
                    kappa * k * temperature / h * np.exp(-96487 * delta_g_b / (R * temperature))
                )

    def as_dict(self) -> dict:
        if self.rate_calculator is None:
            rc = None
        else:
            rc = self.rate_calculator.as_dict()

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": self.r_dicts,
            "products": self.p_dicts,
            "reactant": self.r_dicts[0],
            "product": self.p_dicts[0],
            "inner_reorganization_energy": self.inner_reorganization_energy,
            "dielectric": self.dielectric,
            "refractive": self.refractive,
            "electron_free_energy": self.electron_free_energy,
            "radius": self.radius,
            "electrode_dist": self.electrode_dist,
            "rate_calculator": rc,
            "parameters": self.parameters,
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

        parameters = d["parameters"]

        reaction = cls(
            reactant,
            product,
            d["inner_reorganization_energy"],
            d["dielectric"],
            d["refractive"],
            d["electron_free_energy"],
            d["radius"],
            d["electrode_dist"],
            parameters=parameters,
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
       reactant([MoleculeEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
       transition_state (MoleculeEntry or None): A MoleculeEntry representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
    """

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: MoleculeEntry,
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
    ):
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

        self.class_type = "IntramolSingleBondChangeReaction"
        super().__init__(
            [reactant], [product], transition_state=transition_state, parameters=parameters,
        )

        self.reaction_type(reactant, product)
        self.e0_free_energy_method = reactant.free_energy
        self.e1_free_energy_method = product.free_energy
        self.rct_energy = reactant.energy
        self.pro_energy = product.energy

        self.num_rct = 1
        self.num_pro = 1

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a IntramolSingleBondChangeReaction class object into
        graph representation (nx.Digraph object).
        IntramolSingleBondChangeReaction must be of type 1 reactant -> 1 product

        :return nx.Digraph object of a single IntramolSingleBondChangeReaction object
        """

        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction], Mapping_Family_Dict]:
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
                                                    indices = entry1.mol_graph.extract_bond_environment(
                                                        [edge]
                                                    )
                                                    subg = (
                                                        entry1.graph.subgraph(list(indices))
                                                        .copy()
                                                        .to_undirected()
                                                    )

                                                    families, templates = categorize(
                                                        r, families, templates, subg, charge,
                                                    )
                                                    break

        return reactions, families

    def reaction_type(self, reactant, product):
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
        types = [
            "Intramolecular single bond formation",
            "Intramolecular single bond breakage",
        ]
        self.rxn_type_A = types[int(product.charge < reactant.charge)]
        self.rxn_type_B = types[1 - int(product.charge < reactant.charge)]

    def free_energy(self, temperature=298.15):
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
        if self.e1_free_energy_method is not None and self.e0_free_energy_method is not None:
            self.free_energy_A = self.e1_free_energy_method(
                temp=temperature
            ) - self.e0_free_energy_method(temp=temperature)
            self.free_energy_B = 0 - self.free_energy_A
        else:
            self.free_energy_A = None
            self.free_energy_B = None

    def energy(self):
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

        if self.pro_energy is not None and self.rct_energy is not None:
            self.energy_A = self.pro_energy - self.rct_energy
            self.energy_B = 0 - self.energy_A

        else:
            self.energy_A = None
            self.energy_B = None

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            self.free_energy(temperature=temperature)

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if gb < 0:
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
            "reactants": self.r_dicts,
            "products": self.p_dicts,
            "reactant": self.r_dicts[0],
            "product": self.p_dicts[0],
            "transition_state": ts,
            "rate_calculator": rc,
            "parameters": self.parameters,
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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, product, transition_state=ts, parameters=parameters)
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

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
    ):
        """
          Initializes IntermolecularReaction.reactant to be in the form of a
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

        product_0 = product[0]
        product_1 = product[1]
        self.class_type = "IntermolecularReaction"
        super().__init__(
            [reactant], product, transition_state=transition_state, parameters=parameters,
        )
        self.reaction_type()
        self.rct_free_energy = reactant.get_free_energy
        self.pro0_free_energy = product_0.get_free_energy
        self.pro1_free_energy = product_1.get_free_energy
        self.rct_energy = reactant.energy
        self.pro0_energy = product_0.energy
        self.pro1_energy = product_1.energy
        self.rct_charge = reactant.charge
        self.pro0_charge = product_0.charge
        self.pro1_charge = product_1.charge
        self.rct_formula = reactant.formula
        self.pro0_formula = product_0.formula
        self.pro1_formula = product_1.formula

    def graph_representation(self) -> nx.DiGraph:

        """
        A method to convert a IntermolecularReaction class object into graph
        representation (nx.Digraph object).
        IntermolecularReaction must be of type 1 reactant -> 2 products

        :return nx.Digraph object of a single IntermolecularReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction], Mapping_Family_Dict]:
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
                                    if formula0 in entries and formula1 in entries:
                                        if (
                                            Nbonds0 in entries[formula0]
                                            and Nbonds1 in entries[formula1]
                                        ):
                                            for charge0 in entries[formula0][Nbonds0]:
                                                for entry0 in entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in entries[formula1][Nbonds1]:
                                                            for entry1 in entries[formula1][
                                                                Nbonds1
                                                            ][charge1]:
                                                                if frags[1].isomorphic_to(
                                                                    entry1.mol_graph
                                                                ):
                                                                    r = cls(
                                                                        entry, [entry0, entry1,],
                                                                    )
                                                                    mg = entry.mol_graph
                                                                    indices = mg.extract_bond_environment(
                                                                        [edge]
                                                                    )
                                                                    subg = mg.graph.subgraph(
                                                                        list(indices)
                                                                    ).copy()
                                                                    subg = subg.to_undirected()

                                                                    (
                                                                        families,
                                                                        templates,
                                                                    ) = categorize(
                                                                        r,
                                                                        families,
                                                                        templates,
                                                                        subg,
                                                                        charge,
                                                                    )
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions, families

    def reaction_type(self):

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
        self.rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        self.rxn_type_B = "Molecular formation from one new bond A+B -> C"

    def free_energy(self, temperature=298.15):
        """
        A method to determine the free energy of the intermolecular reaction

        Args:
           :return dictionary of the form {"free_energy_A": energy_A,
                                           "free_energy_B": energy_B}
           where free_energy_A is the primary type of the reaction based on
           the reactant and product of the IntermolecularReaction
           object, and the backwards of this reaction would be free_energy_B.
        """
        if (
            self.pro1_free_energy() is not None
            and self.pro0_free_energy() is not None
            and self.rct_free_energy() is not None
        ):
            self.free_energy_A = (
                self.pro0_free_energy() + self.pro1_free_energy() - self.rct_free_energy()
            )
            self.free_energy_B = 0 - self.free_energy_A

        else:
            self.free_energy_A = None
            self.free_energy_B = None

    def energy(self):
        """
        A method to determine the energy of the intermolecular reaction

        Args:
           :return dictionary of the form {"energy_A": energy_A,
                                           "energy_B": energy_B}
           where energy_A is the primary type of the reaction based on the
           reactant and product of the IntermolecularReaction
           object, and the backwards of this reaction would be energy_B.
        """
        if (
            self.pro1_energy is not None
            and self.pro0_energy is not None
            and self.rct_energy is not None
        ):
            self.energy_A = self.pro0_energy + self.pro1_energy - self.rct_energy
            self.energy_B = 0 - self.energy_A

        else:
            self.energy_A = None
            self.energy_B = None

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            self.free_energy(temperature=temperature)

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if gb < 0:
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
            "reactants": self.r_dicts,
            "products": self.p_dicts,
            "reactant": self.r_dicts[0],
            "product_0": self.p_dicts[0],
            "product_1": self.p_dicts[1],
            "transition_state": ts,
            "rate_calculator": rc,
            "parameters": self.parameters,
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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, [product_0, product_1], transition_state=ts, parameters=parameters)
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

    def __init__(
        self,
        reactant: MoleculeEntry,
        product: List[MoleculeEntry],
        transition_state: Optional[MoleculeEntry] = None,
        parameters: Optional[Dict] = None,
    ):
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
        self.class_type = "CoordinationBondChangeReaction"
        super().__init__(
            [reactant], product, transition_state=transition_state, parameters=parameters,
        )
        self.reaction_type()
        self.rct_free_energy = reactant.get_free_energy
        self.pro0_free_energy = product[0].free_energy
        self.pro1_free_energy = product[1].free_energy

        self.pro1_energy = product[1].energy
        self.pro0_energy = product[0].energy
        self.rct_energy = reactant.energy

    def graph_representation(self) -> nx.DiGraph:
        """
        A method to convert a CoordinationBondChangeReaction class object
            into graph representation (nx.Digraph object).
        CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

        :return nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction], Mapping_Family_Dict]:
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
                    assert len(entries[formula][0][charge]) == 1
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
                                        if (
                                            str(entry.molecule.sites[edge[0]].species) in M_entries
                                            or str(entry.molecule.sites[edge[1]].species)
                                            in M_entries
                                        ):
                                            M_bond = (edge[0], edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs(
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
                                                frag_formula = (
                                                    frag.molecule.composition.alphabetical_formula
                                                )
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
                                                                for nonM_charge in entries[
                                                                    nonM_formula
                                                                ][nonM_Nbonds]:
                                                                    M_charge = (
                                                                        entry.charge - nonM_charge
                                                                    )
                                                                    if (
                                                                        M_charge
                                                                        in M_entries[M_formula]
                                                                    ):
                                                                        for nonM_entry in entries[
                                                                            nonM_formula
                                                                        ][nonM_Nbonds][nonM_charge]:
                                                                            if frag.isomorphic_to(
                                                                                nonM_entry.mol_graph
                                                                            ):
                                                                                this_m = M_entries[
                                                                                    M_formula
                                                                                ][M_charge]
                                                                                r = cls(
                                                                                    entry,
                                                                                    [
                                                                                        nonM_entry,
                                                                                        this_m,
                                                                                    ],
                                                                                )

                                                                                mg = entry.mol_graph

                                                                                indices = mg.extract_bond_environment(
                                                                                    list(bond_pair)
                                                                                )
                                                                                subg = (
                                                                                    mg.graph.subgraph(
                                                                                        list(
                                                                                            indices
                                                                                        )
                                                                                    )
                                                                                    .copy()
                                                                                    .to_undirected()
                                                                                )

                                                                                (
                                                                                    fam,
                                                                                    temp,
                                                                                ) = categorize(
                                                                                    r,
                                                                                    fam,
                                                                                    temp,
                                                                                    subg,
                                                                                    charge,
                                                                                )

                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions, fam

    def reaction_type(self):
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

        self.rxn_type_A = "Coordination bond breaking AM -> A+M"
        self.rxn_type_B = "Coordination bond forming A+M -> AM"

    def free_energy(self, temperature=298.15):
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
        if (
            self.pro1_free_energy() is not None
            and self.pro0_free_energy() is not None
            and self.rct_free_energy() is not None
        ):
            self.free_energy_A = (
                self.pro0_free_energy() + self.pro1_free_energy() - self.rct_free_energy()
            )
            self.free_energy_B = 0 - self.free_energy_A

        else:
            self.free_energy_A = None
            self.free_energy_B = None

    def energy(self):
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
        if (
            self.pro1_energy is not None
            and self.pro0_energy is not None
            and self.rct_energy is not None
        ):
            self.energy_A = self.pro0_energy + self.pro1_energy - self.rct_energy
            self.energy_B = -self.energy_A

        else:
            self.energy_A = None
            self.energy_B = None

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            self.free_energy()

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if gb < 0:
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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
        else:
            ts = MoleculeEntry.from_dict(d["transition_state"])
            rate_calculator = ReactionRateCalculator.from_dict(d["rate_calculator"])

        parameters = d["parameters"]

        reaction = cls(reactant, [product_0, product_1], transition_state=ts, parameters=parameters)
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

        # self.reactants = reactant
        # self.products = product
        self.electron_free_energy = electron_free_energy
        self.electron_energy = None
        self.class_type = "ConcertedReaction"
        super().__init__(
            reactant, product, transition_state=transition_state, parameters=parameters
        )
        self.reaction_type()
        self.rct_free_energies = [r.get_free_energy for r in reactant]
        self.pro_free_energies = [p.get_free_energy for p in product]
        self.rct_energies = [r.energy for r in reactant]
        self.pro_energies = [p.energy for p in product]
        self.rct_charges = [r.charge for r in reactant]
        self.pro_charges = [p.charge for p in product]

    def graph_representation(self,) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

        """
        A method to convert a Concerted class object into graph
            representation (nx.Digraph object).
        IntermolecularReaction must be of type 1 reactant -> 2 products
        :return nx.Digraph object of a single IntermolecularReaction object
        """
        if len(self.rct_indices) == len(self.pro_indices) == 1:
            return graph_rep_1_1(self)
        elif len(self.rct_indices) == 1 and len(self.pro_indices) == 2:
            return graph_rep_1_2(self)
        elif len(self.rct_indices) == 2 and len(self.pro_indices) == 1:
            self.rct_indices, self.pro_indices = self.pro_indices, self.rct_indices
            self.reactant_ids, self.product_ids = self.product_ids, self.reactant_ids
            return graph_rep_1_2(self)
        elif len(self.rct_indices) == len(self.pro_indices) == 2:
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
            from pymatgen.reaction_network.extract_reactions import FindConcertedReactions

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

    def reaction_type(self):

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

        self.rxn_type_A = self.rxn_type_B = "Concerted"

    def free_energy(self, temperature=298.15):
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

        cond_rct = all(nrg() is not None for nrg in self.rct_free_energies)
        cond_pro = all(nrg() is not None for nrg in self.pro_free_energies)
        if cond_rct and cond_pro:
            reactant_charge = np.sum([item for item in self.rct_charges])
            product_charge = np.sum([item for item in self.pro_charges])
            reactant_free_energy = np.sum([item() for item in self.rct_free_energies])
            product_free_energy = np.sum([item() for item in self.pro_free_energies])
            total_charge_change = product_charge - reactant_charge
            free_energy_A = (
                product_free_energy - reactant_free_energy + total_charge_change * electron_free
            )
            free_energy_B = (
                reactant_free_energy - product_free_energy - total_charge_change * electron_free
            )

        else:
            free_energy_A = None
            free_energy_B = None
        self.free_energy_A = free_energy_A
        self.free_energy_B = free_energy_B

    def energy(self):
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
        if all(energy is None for energy in self.rct_energies) and all(
            energy is None for energy in self.pro_energies
        ):
            reactant_total_charge = np.sum([charge for item in self.rct_charges])
            product_total_charge = np.sum([charge for item in self.pro_charges])
            reactant_total_energy = np.sum([energy for item in self.rct_energies])
            product_total_energy = np.sum([energy for item in self.pro_energies])
            # total_charge_change = product_total_charge - reactant_total_charge
            energy_A = product_total_energy - reactant_total_energy
            energy_B = reactant_total_energy - product_total_energy

        else:
            energy_A = None
            energy_B = None
        self.energy_A = energy_A
        self.energy_B = energy_B

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        if isinstance(self.rate_calculator, ReactionRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        elif isinstance(self.rate_calculator, ExpandedBEPRateCalculator):
            return {
                "k_A": self.rate_calculator.calculate_rate_constant(temperature=temperature),
                "k_B": self.rate_calculator.calculate_rate_constant(
                    temperature=temperature, reverse=True
                ),
            }
        else:
            rate_constant = dict()
            self.free_energy()

            ga = self.free_energy_A
            gb = self.free_energy_B

            if ga < 0:
                rate_constant["k_A"] = k * temperature / h
            else:
                rate_constant["k_A"] = (
                    k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))
                )

            if gb < 0:
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
                rate_calculator = ExpandedBEPRateCalculator.from_dict(d["rate_calculator"])
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

    if prod0_index <= prod1_index:
        two_prod_name = str(prod0_index) + "+" + str(prod1_index)
        two_prod_name_entry_ids = str(reaction.product_ids[0]) + "+" + str(reaction.product_ids[1])
    else:
        two_prod_name = str(prod1_index) + "+" + str(prod0_index)
        two_prod_name_entry_ids = str(reaction.product_ids[1]) + "+" + str(reaction.product_ids[0])

    reactants_ind_list = np.array([rct0_index, rct1_index, rct2_index,])
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

    two_prod_name0 = str(prod0_index) + "+PR_" + str(prod1_index)
    two_prod_name1 = str(prod1_index) + "+PR_" + str(prod0_index)

    if rct1_index <= rct2_index:
        three_reac_name0 = str(rct0_index) + "+PR_" + str(rct1_index) + "+PR_" + str(rct2_index)
        three_reac_entry_ids0 = (
            str(reaction.reactant_ids[0])
            + "+PR_"
            + str(reaction.reactant_ids[1])
            + "+PR_"
            + str(reaction.reactant_ids[2])
        )
    else:
        three_reac_name0 = str(rct0_index) + "+PR_" + str(rct2_index) + "+PR_" + str(rct1_index)
        three_reac_entry_ids0 = (
            str(reaction.reactant_ids[0])
            + "+PR_"
            + str(reaction.reactant_ids[2])
            + "+PR_"
            + str(reaction.reactant_ids[1])
        )
    if rct0_index <= rct2_index:
        three_reac_name1 = str(rct1_index) + "+PR_" + str(rct0_index) + "+PR_" + str(rct2_index)
        three_reac_entry_ids1 = (
            str(reaction.reactant_ids[1])
            + "+PR_"
            + str(reaction.reactant_ids[0])
            + "+PR_"
            + str(reaction.reactant_ids[2])
        )
    else:
        three_reac_name1 = str(rct1_index) + "+PR_" + str(rct2_index) + "+PR_" + str(rct0_index)
        three_reac_entry_ids1 = (
            str(reaction.reactant_ids[1])
            + "+PR_"
            + str(reaction.reactant_ids[2])
            + "+PR_"
            + str(reaction.reactant_ids[0])
        )
    if rct0_index <= rct1_index:
        three_reac_name2 = str(rct2_index) + "+PR_" + str(rct0_index) + "+PR_" + str(rct1_index)
        three_reac_entry_ids2 = (
            str(reaction.reactant_ids[2])
            + "+PR_"
            + str(reaction.reactant_ids[0])
            + "+PR_"
            + str(reaction.reactant_ids[1])
        )
    else:
        three_reac_name2 = str(rct2_index) + "+PR_" + str(rct1_index) + "+PR_" + str(rct0_index)
        three_reac_entry_ids2 = (
            str(reaction.reactant_ids[2])
            + "+PR_"
            + str(reaction.reactant_ids[1])
            + "+PR_"
            + str(reaction.reactant_ids[0])
        )

    node_name_A0 = three_reac_name0 + "," + two_prod_name
    node_name_A1 = three_reac_name1 + "," + two_prod_name
    node_name_A2 = three_reac_name2 + "," + two_prod_name
    node_name_B0 = two_prod_name0 + "," + reactants_name
    node_name_B1 = two_prod_name1 + "," + reactants_name

    two_prod_entry_ids0 = str(reaction.product_ids[0]) + "+PR_" + str(reaction.product_ids[1])
    two_prod_entry_ids1 = str(reaction.product_ids[1]) + "+PR_" + str(reaction.product_ids[0])

    entry_ids_name_A0 = three_reac_entry_ids0 + "," + two_prod_name_entry_ids
    entry_ids_name_A1 = three_reac_entry_ids1 + "," + two_prod_name_entry_ids
    entry_ids_name_A2 = three_reac_entry_ids2 + "," + two_prod_name_entry_ids
    entry_ids_name_B0 = two_prod_entry_ids0 + "," + reactants_name_entry_ids
    entry_ids_name_B1 = two_prod_entry_ids1 + "," + reactants_name_entry_ids

    rxn_type_A = reaction.rxn_type_A
    rxn_type_B = reaction.rxn_type_B
    reaction.free_energy()
    free_energy_A = reaction.free_energy_A
    free_energy_B = reaction.free_energy_B
    reaction.energy()
    energy_A = reaction.energy_A
    energy_B = reaction.energy_B

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
        node_name_A0, product_0.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_A0, product_1.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
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
        node_name_A1, product_0.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_A1, product_1.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
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
        node_name_A1, product_0.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_A1, product_1.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
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
        node_name_B0, reactant_0.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B0, reactant_1.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B0, reactant_2.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
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
        node_name_B1, reactant_0.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B1, reactant_1.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B1, reactant_2.parameters["ind"], softplus=0.0, exponent=0.0, weight=1.0,
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

    if len(reaction.rct_indices) != 2 or len(reaction.rct_indices) != 2:
        raise ValueError("Must provide reaction with 2 reactants and 2 products for graph_rep_2_2")

    prod0_index = reaction.pro_indices[0]
    prod1_index = reaction.pro_indices[1]
    rct0_index = reaction.rct_indices[0]
    rct1_index = reaction.rct_indices[1]
    graph = nx.DiGraph()

    if prod0_index <= prod1_index:
        two_prod_name = str(prod0_index) + "+" + str(prod1_index)
        two_prod_name_entry_ids = str(reaction.product_ids[0]) + "+" + str(reaction.product_ids[1])
    else:
        two_prod_name = str(prod1_index) + "+" + str(prod0_index)
        two_prod_name_entry_ids = str(reaction.product_ids[1]) + "+" + str(reaction.product_ids[0])

    if rct0_index <= rct1_index:
        two_reac_name = str(rct0_index) + "+" + str(rct1_index)
        two_reac_name_entry_ids = (
            str(reaction.reactant_ids[0]) + "+" + str(reaction.reactant_ids[1])
        )
    else:
        two_reac_name = str(rct1_index) + "+" + str(rct0_index)
        two_reac_name_entry_ids = (
            str(reaction.reactant_ids[1]) + "+" + str(reaction.reactant_ids[0])
        )

    two_prod_name0 = str(prod0_index) + "+PR_" + str(prod1_index)
    two_prod_name1 = str(prod1_index) + "+PR_" + str(prod0_index)

    two_reac_name0 = str(rct0_index) + "+PR_" + str(rct1_index)
    two_reac_name1 = str(rct1_index) + "+PR_" + str(rct0_index)

    node_name_A0 = two_reac_name0 + "," + two_prod_name
    node_name_A1 = two_reac_name1 + "," + two_prod_name
    node_name_B0 = two_prod_name0 + "," + two_reac_name
    node_name_B1 = two_prod_name1 + "," + two_reac_name

    two_prod_entry_ids0 = str(reaction.product_ids[0]) + "+PR_" + str(reaction.product_ids[1])
    two_prod_entry_ids1 = str(reaction.product_ids[1]) + "+PR_" + str(reaction.product_ids[0])

    two_reac_entry_ids0 = str(reaction.reactant_ids[0]) + "+PR_" + str(reaction.reactant_ids[1])
    two_reac_entry_ids1 = str(reaction.reactant_ids[1]) + "+PR_" + str(reaction.reactant_ids[0])

    entry_ids_name_A0 = two_reac_entry_ids0 + "," + two_prod_name_entry_ids
    entry_ids_name_A1 = two_reac_entry_ids1 + "," + two_prod_name_entry_ids
    entry_ids_name_B0 = two_prod_entry_ids0 + "," + two_reac_name_entry_ids
    entry_ids_name_B1 = two_prod_entry_ids1 + "," + two_reac_name_entry_ids

    rxn_type_A = reaction.rxn_type_A
    rxn_type_B = reaction.rxn_type_B
    reaction.energy()
    energy_A = reaction.energy_A
    energy_B = reaction.energy_B
    reaction.free_energy()
    free_energy_A = reaction.free_energy_A
    free_energy_B = reaction.free_energy_B

    graph.add_node(
        node_name_A0,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A0,
    )

    graph.add_edge(
        rct0_index,
        node_name_A0,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A0, rct0_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_A0, prod0_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
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
        rct1_index,
        node_name_A1,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A1, prod0_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_A1, prod1_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
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
        prod0_index,
        node_name_B0,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )

    graph.add_edge(
        node_name_B0, rct0_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B0, rct1_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
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
        prod1_index,
        node_name_B1,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )

    graph.add_edge(
        node_name_B1, rct0_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B1, rct1_index, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
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
    if len(reaction.rct_indices) != 1 or len(reaction.pro_indices) != 2:
        raise ValueError("Must provide reaction with 1 reactant and 2 products" "for graph_rep_1_2")

    reactant_params = reaction.rct_indices[0]
    reactant_eid = str(reaction.reactant_ids[0])
    product0_params = reaction.pro_indices[0]
    product0_eid = str(reaction.product_ids[0])
    product1_params = reaction.pro_indices[1]
    product1_eid = str(reaction.product_ids[1])

    graph = nx.DiGraph()

    if product0_params <= product1_params:
        two_mol_name = str(product0_params) + "+" + str(product1_params)
        two_mol_name_entry_ids = str(product0_eid) + "+" + str(product1_eid)
    else:
        two_mol_name = str(product1_params) + "+" + str(product0_params)
        two_mol_name_entry_ids = str(product1_eid) + "+" + str(product0_eid)

    two_mol_name0 = str(product0_params) + "+PR_" + str(product1_params)
    two_mol_name1 = str(product1_params) + "+PR_" + str(product0_params)
    node_name_A = str(reactant_params) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(reactant_params)
    node_name_B1 = two_mol_name1 + "," + str(reactant_params)

    two_mol_entry_ids0 = str(product0_eid) + "+PR_" + str(product1_eid)
    two_mol_entry_ids1 = str(product1_eid) + "+PR_" + str(product0_eid)
    entry_ids_name_A = str(reactant_eid) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(reactant_eid)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(reactant_eid)

    rxn_type_A = reaction.rxn_type_A
    rxn_type_B = reaction.rxn_type_B
    reaction.energy()
    energy_A = reaction.energy_A
    energy_B = reaction.energy_B
    reaction.free_energy()
    free_energy_A = reaction.free_energy_A
    free_energy_B = reaction.free_energy_B

    graph.add_node(
        node_name_A,
        rxn_type=rxn_type_A,
        bipartite=1,
        energy=energy_A,
        free_energy=free_energy_A,
        entry_ids=entry_ids_name_A,
    )

    graph.add_edge(
        reactant_params,
        node_name_A,
        softplus=softplus(free_energy_A),
        exponent=exponent(free_energy_A),
        rexp=rexp(free_energy_A),
        weight=1.0,
    )

    graph.add_edge(
        node_name_A, product0_params, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_A, product1_params, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
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
        node_name_B0, reactant_params, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )
    graph.add_edge(
        node_name_B1, reactant_params, softplus=0.0, exponent=0.0, rexp=0.0, weight=1.0,
    )

    graph.add_edge(
        product0_params,
        node_name_B0,
        softplus=softplus(free_energy_B),
        exponent=exponent(free_energy_B),
        rexp=rexp(free_energy_B),
        weight=1.0,
    )
    graph.add_edge(
        product1_params,
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

    if len(reaction.rct_indices) != 1 or len(reaction.pro_indices) != 1:
        raise ValueError("Must provide reaction with 1 reactant and product" "for graph_rep_1_1")

    graph = nx.DiGraph()
    # Reactant Data:
    reactant_params = reaction.rct_indices[0]
    reactant_eid = str(reaction.reactant_ids[0])

    # Product Data:
    product_params = reaction.pro_indices[0]
    product_eid = str(reaction.product_ids[0])

    # Param Updates
    reaction.free_energy()
    reaction.energy()

    def generate_graph_rep(element):
        # Compute element specific attributes
        node_name = (
            str(reactant_params) + "," + str(product_params)
            if element == "A"
            else str(product_params) + "," + str(reactant_params)
        )
        entry_ids = (
            reactant_eid + "," + product_eid if element == "A" else product_eid + "," + reactant_eid
        )
        free_NRG = getattr(reaction, "free_energy_" + element)
        nrg = getattr(reaction, "energy_" + element)

        # Add the node for the reactant/product we are considering
        graph.add_node(
            node_name,
            rxn_type=getattr(reaction, "rxn_type_" + element),
            bipartite=1,
            energy=nrg,
            free_energy=free_NRG,
            entry_ids=entry_ids,
        )

        # Add relevant edges
        graph.add_edge(
            reactant_params if element == "A" else product_params,
            node_name,
            softplus=softplus(free_NRG),
            exponent=exponent(free_NRG),
            rexp=rexp(free_NRG),
            weight=1.0,
        )
        graph.add_edge(
            node_name,
            product_params if element == "A" else reactant_params,
            softplus=0.0,
            exponent=0.0,
            rexp=0.0,
            weight=1.0,
        )

    # Actually generate reps
    generate_graph_rep("A")
    generate_graph_rep("B")
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

    for e, template in enumerate(templates):
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
