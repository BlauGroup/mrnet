import copy
import heapq
import itertools
import time as time
from typing import Dict, List, Tuple, Union, Any, FrozenSet, Set

import networkx as nx
from monty.json import MSONable
from networkx.readwrite import json_graph

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import (
    ConcertedReaction,
    CoordinationBondChangeReaction,
    IntermolecularReaction,
    IntramolSingleBondChangeReaction,
    Reaction,
    RedoxReaction,
    exponent,
    general_graph_rep,
    rexp,
    softplus,
    MetalHopReaction,
)
from mrnet.utils.classes import load_class

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


Mapping_Record_Dict = Dict[int, List[str]]


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
        self.path_dict = {
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
        }

    @property
    def as_dict(self) -> dict:
        """
            A method to convert ReactionPath objection into a dictionary
        :return: d: dictionary containing all te ReactionPath attributes
        """
        d = {
            "@module": self.__class__.__module__,
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
            "path_dict": self.path_dict,
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
    def characterize_path(
        cls,
        path: List[Union[str, int]],
        weight: str,
        min_cost: Dict[int, float],
        graph: nx.DiGraph,
        old_solved_PRs=[],
        PR_byproduct_dict={},
        actualPRs={},
    ):  # -> ReactionPath
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
            pool = list()  # type: List[int]
            pool.append(int(path[0]))
            for ii, step in enumerate(path):
                if ii != len(path) - 1:
                    class_instance.cost += graph[step][path[ii + 1]][weight]
                    if isinstance(step, str):
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
                            if PR_b2 is None:
                                if PR_b in pool_modified:
                                    if PR_b in list(min_cost.keys()):
                                        class_instance.cost = (
                                            class_instance.cost - min_cost[PR_b]
                                        )
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
                                        PR_b_byproducts = PR_byproduct_dict[PR_b][
                                            "byproducts"
                                        ]
                                        start = int(PR_byproduct_dict[PR_b]["start"])
                                        if a in PR_b_byproducts:
                                            # print("path replacement happenning")
                                            new_path_piece1 = actualPRs[PR_b][
                                                start
                                            ].path
                                            new_path_piece2 = [
                                                str(PR_b)
                                                + "+"
                                                + "PR_"
                                                + str(a)
                                                + ","
                                                + str(c)
                                            ]
                                            if concerted:
                                                new_path_piece2 = [
                                                    str(PR_b)
                                                    + "+"
                                                    + "PR_"
                                                    + str(a)
                                                    + ","
                                                    + str(c)
                                                    + "+"
                                                    + str(d)
                                                ]
                                            new_path_piece3 = path[ii + 1 : :]
                                            new_path = (
                                                new_path_piece1
                                                + new_path_piece2
                                                + new_path_piece3
                                            )
                                            # print(path, new_path_piece1, new_path_piece2,new_path_piece3 )
                                            assert (
                                                c == path[ii + 1] or d == path[ii + 1]
                                            )
                                            if new_path_piece2[0] not in graph.nodes:
                                                pool.remove(a)
                                                pool = pool + PR_b_byproducts
                                                pool.append(c)
                                                if concerted:
                                                    pool.append(d)
                                            else:
                                                return ReactionPath.characterize_path(
                                                    new_path,
                                                    weight,
                                                    min_cost,
                                                    graph,
                                                    old_solved_PRs,
                                                    PR_byproduct_dict,
                                                    actualPRs,
                                                )
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
                                    class_instance.cost = (
                                        class_instance.cost - min_cost[PR_b]
                                    )
                                    class_instance.cost = (
                                        class_instance.cost - min_cost[PR_b2]
                                    )
                                    pool.remove(a)
                                    pool.remove(PR_b)
                                    pool.remove(PR_b2)
                                    pool.append(c)
                                    pool.append(d)

                                elif (
                                    PR_b not in old_solved_PRs
                                    and PR_b2 not in old_solved_PRs
                                ):
                                    class_instance.unsolved_prereqs.append(PR_b)
                                    class_instance.unsolved_prereqs.append(PR_b2)
                                    class_instance.all_prereqs.append(PR_b)
                                    class_instance.all_prereqs.append(PR_b2)
                                    pool.remove(a)
                                    pool.append(c)
                                    pool.append(d)

                                elif (
                                    PR_b not in pool_modified
                                    and PR_b2 not in pool_modified
                                ):
                                    if (
                                        PR_b in old_solved_PRs
                                        and PR_b2 in old_solved_PRs
                                    ):
                                        pool.remove(a)
                                        pool.append(c)
                                        pool.append(d)
                                        class_instance.solved_prereqs.append(PR_b)
                                        class_instance.solved_prereqs.append(PR_b2)
                                        class_instance.all_prereqs.append(PR_b)
                                        class_instance.all_prereqs.append(PR_b2)
                                        PR_b_byproducts = PR_byproduct_dict[PR_b][
                                            "byproducts"
                                        ]
                                        PR_b2_byproducts = PR_byproduct_dict[PR_b2][
                                            "byproducts"
                                        ]
                                        pool = pool + PR_b_byproducts + PR_b2_byproducts

                                    elif (
                                        PR_b not in old_solved_PRs
                                        or PR_b2 not in old_solved_PRs
                                    ):
                                        if PR_b not in old_solved_PRs:
                                            class_instance.unsolved_prereqs.append(PR_b)
                                            class_instance.all_prereqs.append(PR_b)
                                        elif PR_b2 not in old_solved_PRs:
                                            class_instance.unsolved_prereqs.append(
                                                PR_b2
                                            )
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
                                        class_instance.cost = (
                                            class_instance.cost - min_cost[PR_in_pool]
                                        )
                                        pool.remove(PR_in_pool)

                                    elif PR_not_in_pool in old_solved_PRs:
                                        class_instance.unsolved_prereqs.append(
                                            PR_not_in_pool
                                        )
                                        class_instance.all_prereqs.append(
                                            PR_not_in_pool
                                        )
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
            pool.remove(int(path[-1]))
            class_instance.byproducts = pool

            class_instance.path_dict = {
                "byproducts": class_instance.byproducts,
                "unsolved_prereqs": class_instance.unsolved_prereqs,
                "solved_prereqs": class_instance.solved_prereqs,
                "all_prereqs": class_instance.all_prereqs,
                "cost": class_instance.cost,
                "path": class_instance.path,
                "overall_free_energy_change": class_instance.overall_free_energy_change,
                "hardest_step": class_instance.hardest_step,
                "description": class_instance.description,
                "pure_cost": class_instance.pure_cost,
                "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                "full_path": class_instance.full_path,
            }

        return class_instance

    @classmethod
    def characterize_path_final(
        cls,
        path: List[Union[str, int]],
        weight: str,
        min_cost: Dict[int, float],
        graph: nx.DiGraph,
        old_solved_PRs=[],
        PR_byproduct_dict={},
        PR_paths={},
    ):
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
            class_instance = cls.characterize_path(
                path,
                weight,
                min_cost,
                graph,
                old_solved_PRs,
                PR_byproduct_dict,
                PR_paths,
            )
            assert len(class_instance.solved_prereqs) == len(class_instance.all_prereqs)
            assert len(class_instance.unsolved_prereqs) == 0

            PRs_to_join = copy.deepcopy(class_instance.all_prereqs)
            full_path = copy.deepcopy(path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = None
                    PR_min_cost = float("inf")
                    for start in PR_paths[PR]:
                        if PR_paths[PR][start].path is not None:
                            if PR_paths[PR][start].cost < PR_min_cost:
                                PR_min_cost = PR_paths[PR][start].cost
                                PR_path = PR_paths[PR][start]
                    if PR_path:
                        assert len(PR_path.solved_prereqs) == len(PR_path.all_prereqs)
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
                        class_instance.pure_cost += ReactionNetwork.softplus(
                            graph.nodes[step]["free_energy"]
                        )
                    elif weight == "exponent":
                        class_instance.pure_cost += ReactionNetwork.exponent(
                            graph.nodes[step]["free_energy"]
                        )
                    elif weight == "rexp":
                        class_instance.pure_cost += ReactionNetwork.rexp(
                            graph.nodes[step]["free_energy"]
                        )

                    class_instance.overall_free_energy_change += graph.nodes[step][
                        "free_energy"
                    ]

                    if class_instance.description == "":
                        class_instance.description += graph.nodes[step]["rxn_type"]
                    else:
                        class_instance.description += (
                            ", " + graph.nodes[step]["rxn_type"]
                        )

                    if class_instance.hardest_step is None:
                        class_instance.hardest_step = step
                    elif (
                        graph.nodes[step]["free_energy"]
                        > graph.nodes[class_instance.hardest_step]["free_energy"]
                    ):
                        class_instance.hardest_step = step

            class_instance.full_path = full_path

            if class_instance.hardest_step is None:
                class_instance.hardest_step_deltaG = None
            else:
                class_instance.hardest_step_deltaG = graph.nodes[
                    class_instance.hardest_step
                ]["free_energy"]

        class_instance.just_path_bp = []
        for ii, step in enumerate(class_instance.path):
            if isinstance(step, int):
                pass
            elif (
                graph.nodes[step]["rxn_type"]
                == "Molecular decomposition breaking one bond A -> B+C"
            ):
                prods = step.split(",")[1].split("+")
                for p in prods:
                    if int(class_instance.path[ii + 1]) != int(p):
                        class_instance.just_path_bp.append(int(p))

        class_instance.path_dict = {
            "byproducts": class_instance.byproducts,
            "just_path_bp": class_instance.just_path_bp,
            "unsolved_prereqs": class_instance.unsolved_prereqs,
            "solved_prereqs": class_instance.solved_prereqs,
            "all_prereqs": class_instance.all_prereqs,
            "cost": class_instance.cost,
            "path": class_instance.path,
            "overall_free_energy_change": class_instance.overall_free_energy_change,
            "hardest_step": class_instance.hardest_step,
            "description": class_instance.description,
            "pure_cost": class_instance.pure_cost,
            "hardest_step_deltaG": class_instance.hardest_step_deltaG,
            "full_path": class_instance.full_path,
        }

        return class_instance


Mapping_PR_Dict = Dict[int, Dict[int, ReactionPath]]


class ReactionNetwork(MSONable):
    """
    Class to build a reaction network from entries
    """

    def __init__(
        self,
        electron_free_energy,
        temperature,
        solvent_dielectric,
        solvent_refractive_index,
        entries_dict,
        entries_list,
        graph,
        reactions,
        PRs,
        PR_record,
        min_cost,
        num_starts,
    ):
        """
        :param electron_free_energy: Electron free energy (in eV)
        :param temperature: Temperature of the system, used for free energy
            and rate constants (temperature given in K)
        :param solvent_dielectric: dielectric constant of the solvent medium
        :param solvent_refractive_index: refractive index of the solvent medium
        :param entries_dict: dict of dicts of dicts of lists (d[formula][bonds][charge])
        :param entries_list: list of unique entries in entries_dict
        :param graph: nx.DiGraph representing connections in the network
        :param reactions: list of Reaction objects
        :param PRs: dict containing prerequisite information
        :param PR_record: dict containing reaction prerequisites
        :param min_cost: dict containing costs of entries in the network
        :param num_starts: Number of starting molecules
        """

        self.electron_free_energy = electron_free_energy
        self.temperature = temperature
        self.solvent_dielectric = solvent_dielectric
        self.solvent_refractive_index = solvent_refractive_index

        self.entries = entries_dict
        self.entries_list = entries_list

        self.graph = graph
        self.PR_record = PR_record
        self.reactions = reactions

        self.min_cost = min_cost
        self.num_starts = num_starts

        self.PRs = PRs
        self.reachable_nodes = []
        self.unsolvable_PRs = []
        self.entry_ids = {e.entry_id for e in self.entries_list}
        self.weight = None
        self.Reactant_record = None
        self.min_cost = {}
        self.not_reachable_nodes = []

    @classmethod
    def from_input_entries(
        cls,
        input_entries,
        electron_free_energy=-2.15,
        temperature=298.15,
        solvent_dielectric=18.5,
        solvent_refractive_index=1.415,
    ):
        """
        Generate a ReactionNetwork from a set of MoleculeEntries.

        :param input_entries: list of MoleculeEntries which will make up the
            network
        :param electron_free_energy: float representing the Gibbs free energy
            required to add an electron (in eV)
        :param temperature: Temperature of the system, used for free energy
            and rate constants (in K)
        :param solvent_dielectric: Dielectric constant of the solvent medium
        :param solvent_refractive_index: Refractive index of the solvent medium
        :return:
        """

        entries = dict()
        entries_list = list()

        print(len(input_entries), "input entries")

        # Filter out unconnected entries, aka those that contain distinctly
        # separate molecules which are not connected via a bond
        connected_entries = list()
        for entry in input_entries:
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")

        def get_formula(x):
            return x.formula

        def get_num_bonds(x):
            return x.num_bonds

        def get_charge(x):
            return x.charge

        def get_free_energy(x):
            return x.get_free_energy(temperature=temperature)

        # Sort by formula
        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_num_bonds)
            entries[k1] = dict()
            # Sort by number of bonds
            for k2, g2 in itertools.groupby(sorted_entries_1, get_num_bonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                entries[k1][k2] = dict()
                # Sort by charge
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = sorted(list(g3), key=get_free_energy)
                    if len(sorted_entries_3) > 1:
                        unique = list()
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            # Sort by graph isomorphism, taking the isomorphic
                            # entry with the lowest free energy
                            for ii, Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    if (
                                        entry.get_free_energy() is not None
                                        and Uentry.get_free_energy() is not None
                                    ):
                                        if entry.get_free_energy(
                                            temperature
                                        ) < Uentry.get_free_energy(temperature):
                                            unique[ii] = entry
                                    elif entry.get_free_energy() is not None:
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

        # Add entry indices
        for ii, entry in enumerate(entries_list):
            if "ind" in entry.parameters.keys():
                pass
            else:
                entry.parameters["ind"] = ii

        entries_list = sorted(entries_list, key=lambda x: x.parameters["ind"])

        graph = nx.DiGraph()

        network = cls(
            electron_free_energy,
            temperature,
            solvent_dielectric,
            solvent_refractive_index,
            entries,
            entries_list,
            graph,
            list(),
            dict(),
            dict(),
            dict(),
            0,
        )

        return network

    @staticmethod
    def softplus(free_energy: float) -> float:
        """
        Method to determine edge weight using softplus cost function
        NOTE: This cost function is unphysical and should only be used when
        neither rexp nor exponent allow prerequisite costs to be solved.
        """
        return softplus(free_energy)

    @staticmethod
    def exponent(free_energy: float) -> float:
        """
        Method to determine edge weight using exponent cost function
        """
        return exponent(free_energy)

    @staticmethod
    def rexp(free_energy: float) -> float:
        """
        Method to determine edge weight using exponent(dG/kt) cost function
        """
        return rexp(free_energy)

    def build(
        self,
        reaction_types: Union[Set, FrozenSet] = frozenset(
            {
                "RedoxReaction",
                "IntramolSingleBondChangeReaction",
                "IntermolecularReaction",
                "CoordinationBondChangeReaction",
            }
        ),
        determine_atom_mappings: bool = True,
    ) -> nx.DiGraph:
        """
            A method to build the reaction network graph

        :param reaction_types (set/frozenset): set/frozenset of all the reactions
            class to include while building the graph
        :param determine_atom_mappings (bool): If True (default), create an atom
            mapping between reactants and products in a given reaction
        :return: nx.DiGraph
        """

        print("build() start", time.time())

        # Add molecule nodes
        for entry in self.entries_list:
            self.graph.add_node(entry.parameters["ind"], bipartite=0)

        reaction_classes = [load_class(str(self.__module__), s) for s in reaction_types]

        all_reactions = list()

        # Generate reactions
        for r in reaction_classes:
            reactions = r.generate(
                self.entries, determine_atom_mappings=determine_atom_mappings
            )
            all_reactions.append(reactions)

        all_reactions = [i for i in all_reactions if i]
        self.reactions = list(itertools.chain.from_iterable(all_reactions))

        redox_c = 0
        inter_c = 0
        intra_c = 0
        coord_c = 0

        for ii, r in enumerate(self.reactions):
            r.parameters["ind"] = ii
            if r.__class__.__name__ == "RedoxReaction":
                redox_c += 1
                r.electron_free_energy = self.electron_free_energy
            elif r.__class__.__name__ == "IntramolSingleBondChangeReaction":
                intra_c += 1
            elif r.__class__.__name__ == "IntermolecularReaction":
                inter_c += 1
            elif r.__class__.__name__ == "CoordinationBondChangeReaction":
                coord_c += 1
            self.add_reaction(r.graph_representation())

        print(
            "redox: ",
            redox_c,
            "inter: ",
            inter_c,
            "intra: ",
            intra_c,
            "coord: ",
            coord_c,
        )
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
        PR_record = {}  # type: Mapping_Record_Dict
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
        Reactant_record = {}  # type: Mapping_Record_Dict
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                Reactant_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                non_PR_reactant = node.split(",")[0].split("+PR_")[0]
                Reactant_record[int(non_PR_reactant)].append(node)
        self.Reactant_record = Reactant_record
        return Reactant_record

    def solve_prerequisites(
        self, starts: List[int], weight: str, max_iter=25
    ):  # -> Tuple[Union[Dict[Union[int, Any], dict], Any], Any]:
        """
            A method to solve all of the prerequisites found in
            ReactionNetwork.graph. By solving all PRs, it gives information on
            1. whether a path exists from any of the starts to each other
            molecule node, 2. if so, what is the min cost to reach that node
            from any of the starts, 3. if there is no path from any of the starts
            to a given molecule node, 4. for molecule nodes where the path
            exists, characterize it in the form of ReactionPath
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
        PRs = {}  # type: Dict[int, Dict[int, ReactionPath]]
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        old_attrs = {}  # type: Dict[Tuple[int, str], Dict[str, float]]
        new_attrs = {}  # type: Dict[Tuple[int, str], Dict[str, float]]
        self.weight = weight
        self.num_starts = len(starts)
        self.PR_byproducts = {}  # type: Dict[int, Dict[str, int]]

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
                    PRs[PR][start] = ReactionPath.characterize_path(
                        [start], weight, self.min_cost, self.graph
                    )
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
            cost_from_start = {}  # type: Dict[int, Dict[int, Union[float, str]]]
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = float("inf")
                self.PR_byproducts[PR] = {}
                for start in PRs[PR]:
                    if PRs[PR][start] == {}:
                        cost_from_start[PR][start] = "no_path"
                    elif PRs[PR][start].path is None:
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start].cost
                        if PRs[PR][start].cost < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start].cost
                            self.PR_byproducts[PR]["byproducts"] = PRs[PR][
                                start
                            ].byproducts
                            self.PR_byproducts[PR]["start"] = start
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"
            PRs, cost_from_start, min_cost = self.find_path_cost(
                starts, weight, old_solved_PRs, cost_from_start, min_cost, PRs
            )
            solved_PRs = copy.deepcopy(old_solved_PRs)
            solved_PRs, new_solved_PRs, cost_from_start = self.identify_solved_PRs(
                PRs, solved_PRs, cost_from_start
            )

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

        print(
            "total input molecules:",
            len(self.entries_list),
            "solvable PRs:",
            len(old_solved_PRs),
            "unsolvable PRs:",
            len(self.unsolvable_PRs),
            "not reachable mols:",
            len(self.not_reachable_nodes),
        )
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
                assert "," in step
                nodes = nodes + step.split(",")
        nodes.pop(0)
        if len(nodes) != 0:
            nodes.pop(-1)
        return nodes, PR, Reactants

    def find_path_cost(
        self, starts, weight, old_solved_PRs, cost_from_start, min_cost, PRs
    ):
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
            dist, paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(
                self.graph, start, weight=self.weight
            )
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
                                PR.append(
                                    int(
                                        step.split("+")[1].split("PR_")[1].split(",")[0]
                                    )
                                )
                                if node in PR:
                                    if node not in wrong_paths[start]:
                                        wrong_paths[start].append(int(node))
                                nodes = nodes + step.split("+")[1].split("PR_")[
                                    1
                                ].split(",")
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
                            assert "," in step
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
                    (
                        length,
                        dij_path,
                    ) = nx.algorithms.simple_paths._bidirectional_dijkstra(
                        self.graph,
                        source=hash(start),
                        target=hash(node),
                        ignore_nodes=self.find_or_remove_bad_nodes(
                            [node] + self.not_reachable_nodes
                        ),
                        weight=self.weight,
                    )
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
                    dist_and_path[start][node]["cost"] = fixed_paths[start][node][
                        "cost"
                    ]
                    dist_and_path[start][node]["path"] = fixed_paths[start][node][
                        "path"
                    ]
                    nodes, PR, reactant = self.parse_path(
                        dist_and_path[start][node]["path"]
                    )
                    dist_and_path[start][node]["all_nodes"] = nodes
                    dist_and_path[start][node]["PRs"] = PR
                    dist_and_path[start][node]["reactant"] = reactant
            dist_and_path[start] = {
                key: value
                for key, value in sorted(
                    dist_and_path[start].items(), key=lambda item: int(item[0])
                )
            }

        for start in starts:
            for node in dist_and_path[start]:
                if node not in old_solved_PRs:
                    if dist_and_path[start][node] == {}:
                        PRs[node][start] = ReactionPath(None)
                        cost_from_start[node][start] = "no_path"
                    elif dist_and_path[start][node]["cost"] == float("inf"):
                        PRs[node][start] = ReactionPath(None)
                    else:
                        path_class = ReactionPath.characterize_path(
                            dist_and_path[start][node]["path"],
                            weight,
                            self.min_cost,
                            self.graph,
                            old_solved_PRs,
                            PR_byproduct_dict=self.PR_byproducts,
                            actualPRs=PRs,
                        )
                        cost_from_start[node][start] = path_class.cost
                        if len(path_class.unsolved_prereqs) == 0:
                            PRs[node][start] = path_class
                        if path_class.cost < min_cost[node]:
                            min_cost[node] = path_class.cost
                            self.PR_byproducts[node][
                                "byproducts"
                            ] = path_class.byproducts
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
                    best_start_so_far = [None, float("inf")]
                    for start in PRs[PR]:
                        if PRs[PR][start] is not None:  # ALWAYS TRUE should be != {}
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

    def update_edge_weights(
        self, min_cost: Dict[int, float], orig_graph: nx.DiGraph
    ) -> Dict[Tuple[int, str], Dict[str, float]]:
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
                    self.weight: orig_graph[non_PR_reactant_node][rxn_node][self.weight]
                    + min_cost[PR_ind]
                }

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
                    if PRs[PR][start].cost == float("inf"):
                        PRs[PR][start] = ReactionPath(None)
                    if PRs[PR][start].path is not None:
                        path_found = True
                        path_dict_class = ReactionPath.characterize_path_final(
                            PRs[PR][start].path,
                            self.weight,
                            self.min_cost,
                            self.graph,
                            self.solved_PRs,
                            PR_byproduct_dict=self.PR_byproducts,
                            PR_paths=PRs,
                        )
                        PRs[PR][start] = path_dict_class
                        if (
                            abs(path_dict_class.cost - path_dict_class.pure_cost)
                            > 0.0001
                        ):
                            print(
                                "WARNING: cost mismatch for PR",
                                PR,
                                path_dict_class.cost,
                                path_dict_class.pure_cost,
                                path_dict_class.path_dict,
                                path_dict_class.full_path,
                            )

                if not path_found:
                    print("No path found from any start to PR", PR)
            else:
                self.unsolvable_PRs.append(PR)
                print("Unsolvable path from any start to PR", PR)
        self.PRs = PRs
        return PRs

    def remove_node(self, node_ind):
        """
        Remove a species from self.graph. Also remove all the reaction nodes with that species.
        Used for e.g. removing Li0.
        :param: list of node numbers to remove
        """
        for n in node_ind:
            self.graph.remove_node(n)
            nodes = list(self.graph.nodes)
            for node in nodes:
                if self.graph.nodes[node]["bipartite"] == 1:
                    reactants = node.split(",")[0].split("+")
                    reactants = [reac.replace("PR_", "") for reac in reactants]
                    products = node.split(",")[1].split("+")
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
            self.Product_record.pop(n, None)

    def find_or_remove_bad_nodes(
        self, nodes: List[int], remove_nodes=False
    ) -> Union[List[str], nx.DiGraph]:
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

    def valid_shortest_simple_paths(
        self, start: int, target: int, PRs=[]
    ):  # -> Generator[List[str]]:????
        """
            A method to determine shortest path from start to target
        :param start: molecular node of type int from ReactionNetwork.graph
        :param target: molecular node of type int from ReactionNetwork.graph
        :param PRs: not used currently?
        :return: nx.path_generator of type generator
        """
        valid_graph = self.find_or_remove_bad_nodes([target], remove_nodes=True)
        valid_graph.remove_nodes_from(PRs)  # type: ignore

        return nx.shortest_simple_paths(
            valid_graph, hash(start), hash(target), weight=self.weight
        )

    def find_paths(self, starts, target, weight, num_paths=10, ignorenode=[]):  # -> ??
        """
            A method to find the shorted path from given starts to a target

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
                        path_dict_class2 = ReactionPath.characterize_path_final(
                            path,
                            self.weight,
                            self.min_cost,
                            self.graph,
                            self.solved_PRs,
                            PR_byproduct_dict=self.PR_byproducts,
                            PR_paths=self.PRs,
                        )
                        heapq.heappush(
                            my_heapq, (path_dict_class2.cost, next(c), path_dict_class2)
                        )
        except Exception:
            print("ind", ind)
        top_path_list = []
        while len(paths) < num_paths and my_heapq:
            (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
            top_path_list.append(path_dict_HP_class.path)
            print(
                len(paths),
                cost_HP,
                path_dict_HP_class.overall_free_energy_change,
                path_dict_HP_class.hardest_step_deltaG,
                path_dict_HP_class.path_dict,
            )
            paths.append(
                path_dict_HP_class.path_dict
            )  # ideally just append the class, but for now dict for easy printing

        self.paths = paths
        self.top_path_list = top_path_list
        print("find_paths end", time.time())

        return self.PRs, paths, top_path_list

    @staticmethod
    def mols_w_cuttoff(RN_pr_solved, cutoff=0, build_pruned_network=True):
        """
        A method to identify molecules reached by dG <= cutoff

        :param RN_pr_solved: instance of reaction network
        :param: cutoff: dG value
        :return: mols_to_keep: list of molecule nodes that can be reached by dG <= cutoff
        :return: pruned_entries_list: list of MoleculeEntry of molecules that can be reached by dG <= cutoff
        """

        pruned_PRs = {}
        for PR_node in RN_pr_solved.PR_byproducts:
            if (
                RN_pr_solved.PRs[PR_node] != {}
                and RN_pr_solved.PR_byproducts[PR_node] != {}
            ):
                min_start = RN_pr_solved.PR_byproducts[PR_node]["start"]
                if (
                    RN_pr_solved.PRs[PR_node][min_start].overall_free_energy_change
                    <= cutoff
                ):
                    pruned_PRs[PR_node] = {}
                    pruned_PRs[PR_node][min_start] = RN_pr_solved.PRs[PR_node][
                        min_start
                    ]

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
            pruned_network_build = ReactionNetwork.from_input_entries(
                pruned_entries_list, replace_ind=False
            )
            pruned_network_build.build()
            return mols_to_keep, pruned_entries_list, pruned_network_build
        else:
            return mols_to_keep, pruned_entries_list

    @staticmethod
    def identify_concerted_rxns_via_intermediates(
        RN_pr_solved, mols_to_keep, single_elem_interm_ignore=["C1", "H1", "O1", "Li1"]
    ):
        """
            A method to identify concerted reactions by looping through high enery intermediates
        :param RN_pr_solved: ReactionNetwork that is PR solved
        :param mols_to_keep: List of pruned molecules, if not running then a list of all molecule nodes in the
        RN_pr_solved
        :param single_elem_interm_ignore: List of formula of high energy intermediates to ignore
        :return: list of reactions
        """

        print("identify_concerted_rxns_via_intermediates start", time.time())
        mols_to_keep.append(None)
        count_total = 0
        reactions = []
        not_wanted_formula = single_elem_interm_ignore
        for entry in RN_pr_solved.entries_list:
            node = entry.parameters["ind"]
            if (
                RN_pr_solved.entries_list[node].formula not in not_wanted_formula
                and RN_pr_solved.graph.nodes[node]["bipartite"] == 0
                and node not in RN_pr_solved.not_reachable_nodes
                and node not in RN_pr_solved.unsolvable_PRs
            ):
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
                        total_dG = (
                            rxn1_dG + RN_pr_solved.graph.nodes[out_node]["free_energy"]
                        )
                        if rxn1_dG > 0 and total_dG < 0:
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
                                        reactant = int(in_mol)
                                        product = int(out_mol)
                                        glist = [reactant, product]
                                        if set(glist).issubset(set(mols_to_keep)):
                                            count = count + 1
                                            reactions.append(
                                                (
                                                    [reactant],
                                                    [product],
                                                    [in_node, out_node],
                                                )
                                            )
                                            # print(([reactant], [product]), in_node, out_node)
                                elif "+" in in_node and "+" not in out_node:
                                    reactant = int(in_node.split(",")[0])
                                    product1 = in_node.split(",")[1].split("+")
                                    product1.remove(str(node))
                                    product1 = int(product1[0])
                                    product2 = int(out_node.split(",")[1])
                                    glist = [
                                        int(reactant),
                                        int(product1),
                                        int(product2),
                                    ]
                                    if set(glist).issubset(set(mols_to_keep)):
                                        count = count + 1
                                        reactions.append(
                                            (
                                                [reactant],
                                                [product1, product2],
                                                [in_node, out_node],
                                            )
                                        )
                                elif "+" not in in_node and "+" in out_node:
                                    reactant = int(in_node.split(",")[0])
                                    product1 = int(out_node.split(",")[1].split("+")[0])
                                    product2 = int(out_node.split(",")[1].split("+")[1])
                                    glist = [
                                        int(reactant),
                                        int(product1),
                                        int(product2),
                                    ]
                                    if set(glist).issubset(set(mols_to_keep)):
                                        count = count + 1
                                        reactions.append(
                                            (
                                                [reactant],
                                                [product1, product2],
                                                [in_node, out_node],
                                            )
                                        )
                            elif "PR" in in_node and "PR" not in out_node:
                                if "+" in out_node:
                                    p1 = list(
                                        map(int, out_node.split(",")[1].split("+"))
                                    )[0]
                                    p2 = list(
                                        map(int, out_node.split(",")[1].split("+"))
                                    )[1]
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
                                if set(glist).issubset(set(mols_to_keep)) and (
                                    {reactant1, reactant2} != {product1, product2}
                                ):
                                    count = count + 1
                                    # print(glist, set(glist).issubset(set(mols_to_keep)))
                                    reactions.append(
                                        (
                                            [reactant1, reactant2],
                                            [product1, product2],
                                            [in_node, out_node],
                                        )
                                    )
                            elif "PR" not in in_node and "PR" in out_node:
                                if (
                                    "+" in in_node
                                ):  # want this: 2441,2426''2426+PR_148,3669'
                                    start = in_node.split(",")[0]
                                    p1 = list(
                                        map(int, in_node.split(",")[1].split("+"))
                                    )
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

                                if set(glist).issubset(set(mols_to_keep)) and (
                                    {reactant1, reactant2} != {product1, product2}
                                ):
                                    count = count + 1
                                    # print(glist, set(glist).issubset(set(mols_to_keep)))
                                    reactions.append(
                                        (
                                            [reactant1, reactant2],
                                            [product1, product2],
                                            [in_node, out_node],
                                        )
                                    )
                count_total = count_total + count
                # print(node, entry.formula, count)
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
                if reaction[1][1] is None:
                    new_node = True
                else:
                    assert int(reaction[1][1]) in pruned_network_build.graph.nodes
                if not new_node:
                    reactant_0 = full_network_pr_solved.entries_list[
                        int(reaction[0][0])
                    ]
                    PR = full_network_pr_solved.entries_list[int(reaction[0][1])]
                    product_0 = full_network_pr_solved.entries_list[int(reaction[1][0])]
                    product_1 = full_network_pr_solved.entries_list[int(reaction[1][1])]
                    cr = ConcertedReaction([reactant_0, PR], [product_0, product_1])
                    cr.electron_free_energy = -2.15
                    g = cr.graph_representation()
                    for node in list(g.nodes):
                        if (
                            not isinstance(node, int)
                            and g.nodes[node]["free_energy"] > 0
                        ):
                            g.remove_node(node)
                    pruned_network_build.add_reaction(g)
                    c3 = c3 + 1
                elif new_node:
                    reactant_0 = full_network_pr_solved.entries_list[
                        int(reaction[0][0])
                    ]
                    PR = full_network_pr_solved.entries_list[int(reaction[0][1])]
                    product_0 = full_network_pr_solved.entries_list[int(reaction[1][0])]
                    cr = ConcertedReaction([reactant_0, PR], [product_0])
                    cr.electron_free_energy = -2.15
                    g = cr.graph_representation()
                    for node in list(g.nodes):
                        if (
                            not isinstance(node, int)
                            and g.nodes[node]["free_energy"] > 0
                        ):
                            g.remove_node(node)
                    pruned_network_build.add_reaction(g)
                    c3 = c3 + 1

        pruned_network_build.PR_record = pruned_network_build.build_PR_record()
        pruned_network_build.Reactant_record = (
            pruned_network_build.build_reactant_record()
        )
        print("add_concerted_rxns end", time.time())
        return pruned_network_build

    def as_dict(self) -> dict:
        entries = dict()  # type: Dict[str, Dict[int, Dict[int, List[Dict[str, Any]]]]]
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

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "entries_dict": entries,
            "entries_list": entries_list,
            "reactions": reactions,
            "electron_free_energy": self.electron_free_energy,
            "temperature": self.temperature,
            "solvent_dielectric": self.solvent_dielectric,
            "solvent_refractive_index": self.solvent_refractive_index,
            "graph": json_graph.adjacency_data(self.graph),
            "PR_record": self.PR_record,
            "min_cost": self.min_cost,
            "num_starts": self.num_starts,
            "PRs": self.PRs,
        }

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
                        entries[formula][int_bonds][int_charge].append(
                            MoleculeEntry.from_dict(entry)
                        )

        entries_list = [MoleculeEntry.from_dict(e) for e in d["entries_list"]]

        reactions = list()
        for reaction in d["reactions"]:
            rclass = load_class(str(cls.__module__), reaction["@class"])
            reactions.append(rclass.from_dict(reaction))

        graph = json_graph.adjacency_graph(d["graph"], directed=True)

        return cls(
            d["electron_free_energy"],
            d["temperature"],
            d["solvent_dielectric"],
            d["solvent_refractive_index"],
            entries,
            entries_list,
            graph,
            reactions,
            d.get("PR_record", dict()),
            d.get("PRs", dict()),
            d.get("min_cost", dict()),
            d["num_starts"],
        )
