import copy
import heapq
import os
import itertools
import operator
import pickle
import time as time
from functools import reduce
from typing import Dict, List, Tuple, Union, Any, FrozenSet, Set, TypeVar
from ast import literal_eval
import networkx as nx
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from networkx.readwrite import json_graph

from mrnet.network.reaction_generation import ReactionIterator, EntriesBox
from mrnet.core.mol_entry import MoleculeEntry
from pymatgen.analysis.graphs import MoleculeGraph
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
RN_type = TypeVar("RN_type", bound="ReactionNetwork")


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
        graph: nx.DiGraph,
        old_solved_PRs=[],
    ):  # -> ReactionPath
        """
         A method to define ReactionPath attributes based on the inputs

        :param path: a list of nodes that defines a path from node A to B
            within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param graph: nx.Digraph
        :param old_solved_PRs: previously solved PRs from the iterations before
            the current iteration
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
                    if isinstance(step, str):  # REACTION NODE
                        reactants = step.split(",")[0]
                        # products = step.split(",")[1]
                        if "+" in reactants:  # prs for this reaction
                            prod = []  # type: List[Union[str, int]]
                            a = path[ii - 1]  # source reactant (non-pr)
                            rct_indices = list(reactants.split("+"))
                            rct_indices.remove(str(a))
                            a = int(a)
                            pr = int(rct_indices[0])
                            if "+" in step.split(",")[1]:
                                c = int(step.split(",")[1].split("+")[0])
                                d = int(step.split(",")[1].split("+")[1])
                                prod = [c, d]
                            else:
                                c = int(step.split(",")[1])
                                prod = [c]
                            if pr in old_solved_PRs:
                                class_instance.solved_prereqs.append(pr)
                            else:
                                class_instance.unsolved_prereqs.append(pr)
                            class_instance.all_prereqs.append(pr)
                            pool.remove(a)
                            pool = pool + prod
                        elif "+" in step.split(",")[1]:
                            # node = A,B+C
                            a = int(step.split(",")[0])
                            b = int(step.split(",")[1].split("+")[0])
                            c = int(step.split(",")[1].split("+")[1])
                            pool.remove(a)
                            pool.append(b)
                            pool.append(c)
                        else:
                            # node = A,B
                            a = int(step.split(",")[0])
                            b = int(step.split(",")[1])
                            pool.remove(a)
                            pool.append(b)
        pool.remove(class_instance.path[-1])
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
        graph: nx.DiGraph,
        old_solved_PRs=[],
        PR_paths={},
        PR_byproduct_dict={},
    ):
        """
            A method to define all the attributes of a given path once all the PRs are solved
        :param path: a list of nodes that defines a path from node A to B within a graph built using
        ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
        if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param graph: nx.Digraph
        :param PR_paths: dict that defines a path from each node to a start,
               of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}},
               int(node2):...}
        :return: ReactionPath object
        """

        if path is None:
            class_instance = cls(None)
        else:
            class_instance = cls.characterize_path(path, weight, graph, old_solved_PRs)
            assert len(class_instance.solved_prereqs) == len(class_instance.all_prereqs)
            assert len(class_instance.unsolved_prereqs) == 0

            PRs_to_join = copy.deepcopy(class_instance.all_prereqs)
            full_path = copy.deepcopy(class_instance.path)
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
                    if PR_path is not None:
                        assert len(PR_path.solved_prereqs) == len(PR_path.all_prereqs)
                        for new_PR in PR_path.all_prereqs:
                            new_PRs.append(new_PR)
                        full_path = PR_path.path + full_path
                PRs_to_join = copy.deepcopy(new_PRs)
            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    print("NOTE: Matching prereq and byproduct found!", PR)
                BPs = PR_byproduct_dict[PR]["byproducts"]
                class_instance.byproducts = class_instance.byproducts + BPs

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

    def __eq__(self, obj):
        if type(self) == type(obj):
            return self.as_dict() == obj.as_dict()
        else:
            return False


Mapping_PR_Dict = Dict[int, Dict[int, ReactionPath]]


class ReactionNetwork(MSONable):
    """
    Class to build a reaction network from entries
    """

    def __init__(
        self,
        reaction_iterator,
        electron_free_energy=-2.15,
        temperature=298.15,
        solvent_dielectric=18.5,
        solvent_refractive_index=1.415,
        replace_ind=True,
        reaction_types: Union[Set, FrozenSet] = frozenset(
            {
                "RedoxReaction",
                "IntramolSingleBondChangeReaction",
                "IntermolecularReaction",
                "CoordinationBondChangeReaction",
            }
        ),
        determine_atom_mappings: bool = True,
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
        :param replace_ind: True if reindex the entries if it there is already
            indices in the input_entries
        :return:
        """

        self.electron_free_energy = electron_free_energy
        self.temperature = temperature
        self.solvent_dielectric = solvent_dielectric
        self.solvent_refractive_index = solvent_refractive_index

        # self.entries = reaction_iterator.rn.entries
        self.entries_list = reaction_iterator.entries_box.entries_list

        self.graph = nx.DiGraph()

        self.PRs: dict = dict()
        self.reachable_nodes: list = []
        self.unsolvable_PRs: list = []
        self.entry_ids = {e.entry_id for e in self.entries_list}
        self.min_cost: dict = {}
        self.not_reachable_nodes: list = []

        print("build() start", time.time())

        # Add molecule nodes
        for entry in self.entries_list:
            self.graph.add_node(entry.parameters["ind"], bipartite=0)

        self.reactions = []
        for reaction in reaction_iterator:
            self.reactions.append(reaction)
            self.add_reaction(reaction.graph_representation())

        self.PR_record = self.build_PR_record()  # begin creating PR list
        self.Reactant_record = self.build_reactant_record()  # begin creating rct list

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
        A method to determine all the reaction nodes that have the same
        PR in the ReactionNetwork.graph

        :return: a dict of the form {int(node1): [all the reaction nodes with
        PR of node1, ex "2+node1, 3"]}
        """
        PR_record = {
            int(specie): [] for specie in self.graph.nodes if isinstance(specie, int)
        }  # type: Mapping_Record_Dict
        for edge in filter(lambda e: not isinstance(e[1], int), self.graph.edges()):
            # for edge (u,v), PR is all species in reaction v other than u
            edge_prs = self.graph[edge[0]][edge[1]]["PRs"]
            for pr in edge_prs:
                if pr in PR_record.keys():
                    PR_record[pr].append(edge)
                else:
                    PR_record[pr] = [edge]
        PR_record = {key: list(set(PR_record[key])) for key in PR_record}
        self.PR_record = PR_record
        return PR_record

    def build_reactant_record(self) -> Mapping_Record_Dict:
        """
        A method to determine all the reaction nodes that have the same non
        PR reactant node in the ReactionNetwork.graph

        :return: a dict of the form {int(node1): [all the reaction nodes with
        non PR reactant of node1, ex "node1+2, 3"]}
        """
        Reactant_record = {
            int(specie): [] for specie in self.graph.nodes if isinstance(specie, int)
        }  # type: Mapping_Record_Dict

        # filter to just get weighted edges, then add u of (u,v) to reactant record
        for edge in filter(lambda e: not isinstance(e[1], int), self.graph.edges()):
            # for edge (u,v), PR is all species in reaction v other than u
            Reactant_record[edge[0]].append(edge)
        self.Reactant_record = Reactant_record
        return Reactant_record

    def solve_prerequisites(
        self,
        starts: List[int],
        weight: str,
        max_iter=25,
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

        orig_graph = copy.deepcopy(self.graph)

        for start in starts:  # all the molecular nodes
            PRs[start] = {}  # no PRs necessary @init
        for PR in PRs:  # iter over each PR (eq to molecular nodes) [keys]
            for start in starts:  # iter over molecular nodes
                if start == PR:
                    PRs[PR][start] = ReactionPath.characterize_path(
                        [start], weight, self.graph
                    )
                else:
                    PRs[PR][start] = ReactionPath(None)  # PRs[mol][other_mol]
                    # NO PATH

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
                starts,
                weight,
                old_solved_PRs,
                cost_from_start,
                min_cost,
                PRs,
            )

            solved_PRs = copy.deepcopy(old_solved_PRs)


            solved_PRs, new_solved_PRs, cost_from_start = self.identify_solved_PRs(
                PRs, solved_PRs, cost_from_start
            )

            # print(ii, len(old_solved_PRs), len(new_solved_PRs), new_solved_PRs)
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
            elif "+" in step.split(",")[0]:  # PR
                source = nodes[-1]
                sides = step.split(",")
                if (
                    step.count("+") == 1 or step.count("+") == 2
                ):  # A+B -> C OR A+B+C -> D
                    rct = str(source)
                    nodes = nodes + [rct]
                    Reactants.append(int(rct))
                    pr = [int(el) for el in sides[0].split("+") if el != rct]
                    PR.append(pr)
                    nodes = nodes + [sides[1]]
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
        self,
        starts,
        weight,
        old_solved_PRs,
        cost_from_start,
        min_cost,
        PRs,
        generate=False,
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
                if self.graph.nodes[node]["bipartite"] == 0:  # molecule node
                    if node not in self.reachable_nodes:
                        self.reachable_nodes.append(int(node))

                    dist_and_path[start][int(node)] = {}
                    dist_and_path[start][node]["cost"] = dist[node]
                    dist_and_path[start][node]["path"] = paths[node]
                    nodes = []
                    PR = []
                    Reactants = []
                    for ii, step in enumerate(paths[node]):
                        if isinstance(step, int):
                            nodes.append(step)
                        elif "+" in step.split(",")[0]:  # Has PRs
                            if step.count("+") == 1:
                                nodes = nodes + [step.split("+")[0]]
                                Reactants.append(
                                    int(paths[node][ii - 1])
                                )  # source reactant
                                # "pr" reactant identification
                                source = str(paths[node][ii - 1])
                                rct_indices = list(step.split(",")[0].split("+"))
                                rct_indices.remove(source)
                                PR.append(int(rct_indices[0]))
                                if node in PR:
                                    if node not in wrong_paths[start]:
                                        wrong_paths[start].append(int(node))
                                nodes = nodes + step.split("+")[1].split(",")
                            elif step.count("+") == 2:  # A + PR_B -> C + D
                                nodes = nodes + [step.split(",")[0].split("+")[0]]
                                rcts = step.split(",")[0].split("+")
                                Reactants.append(
                                    int(paths[node][ii - 1])
                                )  # source reactant
                                rcts.remove(
                                    str(paths[node][ii - 1])
                                )  # remove "reactant" reactant
                                assert len(rcts) == 1
                                PR.append(int(rcts[0]))
                                if node in PR:
                                    if node not in wrong_paths[start]:
                                        wrong_paths[start].append(int(node))
                                nodes = nodes + step.split(",")[1].split("+")  # C, D
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
                            self.graph,
                            old_solved_PRs,
                        )
                        if (
                            start == 456
                            and node == 2
                            and path_class.unsolved_prereqs == []
                            and generate
                        ):
                            self.generate_characterize_path_files(
                                old_solved_PRs, dist_and_path[start][node]["path"]
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
        self,
        min_cost: Dict[int, float],
        orig_graph: nx.DiGraph,
    ):
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
        for PR_ind in min_cost:  # all PRs in path
            for weighted_edge in self.PR_record[PR_ind]:  # all edges with this PR
                # split = weighted_edge.split(",")
                # u = split[0]
                # v = split[1]
                attrs[weighted_edge] = {
                    self.weight: orig_graph[weighted_edge[0]][weighted_edge[1]][
                        self.weight
                    ]
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
                            self.graph,
                            self.solved_PRs,
                            PRs,
                            self.PR_byproducts,
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

    def remove_node(self, node_ind: List[int]):
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
                    products = node.split(",")[1].split("+")
                    if str(n) in products:
                        if len(reactants) == 2:
                            other_reac = [int(r) for r in reactants if r != node]
                            assert len(other_reac) == 1
                            self.PR_record[other_reac[0]].remove(node)
                            self.graph.remove_node(node)
                            self.PR_record.pop(node, None)
                    elif str(n) in reactants:
                        if len(reactants) == 2:
                            other_reac = [int(r) for r in reactants if r != node]
                            assert len(other_reac) == 1
                            self.PR_record[other_reac[0]].remove(node)
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
                bad_nodes.append(bad_node[1])
            for bad_nodes2 in self.Reactant_record[node]:
                bad_nodes.append(bad_nodes2[1])
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
                            self.graph,
                            self.solved_PRs,
                            self.PRs,
                            self.PR_byproducts,
                        )
                        heapq.heappush(
                            my_heapq, (path_dict_class2.cost, next(c), path_dict_class2)
                        )
        except Exception:
            print("no path from this start to the target", start)
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
    def parse_reaction_node(node: str):
        """
        A method to identify reactants, PR, and prodcuts from a given reaction node string.
        :param node: string, ex. "1+2,3+4"
        :return: react_list: reactant list, ex [1,2]
        :return: prod_list: product list, ex [3,4]
        """
        react_list_str = node.split(",")[0].split("+")
        prod_list_str = node.split(",")[1].split("+")
        prod_list_str.sort()
        react_list: List[int] = [int(r) for r in react_list_str]
        prod_list: List[int] = [int(p) for p in prod_list_str]
        return (react_list, prod_list)

    @staticmethod
    def generate_node_string(combined_reactants, combined_products):
        """
        A method to genrate a reaction node string from given reactants and products.
        :param combined_reactants: list of reactant node indices, ex [1,2]
        :param combined_products: list of product node indices, ex [3,4]
        :return: node_str: string of reaction as it would be for a reaction node, ex  "1+2,3+4"
        """
        combined_reactants = list(map(str, combined_reactants))
        node_str = (
            "+".join(list(map(str, combined_reactants)))
            + ","
            + "+".join(list(map(str, combined_products)))
        )
        return node_str


def path_finding_wrapper(
    mol_list: List[MoleculeEntry], init_mols: List[MoleculeEntry], target: MoleculeEntry
):
    entries_box = EntriesBox(mol_list)
    ri = ReactionIterator(entries_box)
    rn = ReactionNetwork(ri)

    initial_inds = [e.parameters["ind"] for e in init_mols]

    rn.solve_prerequisites(initial_inds, "softplus")

    # set initial conditions
    PRs, paths, top_path_list = rn.find_paths(
        initial_inds, target.parameters["ind"], weight="softplus", num_paths=20
    )
    # return shortest paths to every mol
    return PRs
