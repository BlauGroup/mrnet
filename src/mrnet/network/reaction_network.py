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

from mrnet.utils.visualization import (
    visualize_molecules,
    generate_latex_header,
    generate_latex_footer,
    latex_emit_reaction,
)

from mrnet.network.reaction_path import ReactionPath
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
    default_cost,
    MetalHopReaction,
)
from mrnet.utils.classes import load_class

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"

Mapping_Record_Dict = Dict[int, List[str]]
RN_type = TypeVar("RN_type", bound="ReactionNetwork")
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
        add_concerteds=True,
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

        self.electron_free_energy = electron_free_energy
        self.temperature = temperature
        self.solvent_dielectric = solvent_dielectric
        self.solvent_refractive_index = solvent_refractive_index
        self.add_concerteds = add_concerteds

        self.entries_list = reaction_iterator.entries_box.entries_list
        self.graph = nx.DiGraph()
        self.PRs: dict = dict()
        self.reachable_nodes: list = []
        self.unsolvable_PRs: list = []
        self.entry_ids = {e.entry_id for e in self.entries_list}
        self.min_cost: dict = {}
        self.not_reachable_nodes: list = []

        print("init() start", time.time())

        # Add molecule nodes
        for entry in self.entries_list:
            self.graph.add_node(entry.parameters["ind"], bipartite=0)

        count = 0
        for reaction in reaction_iterator:
            if reaction_iterator.intermediate_index == -1:
                reaction_object = reaction_iterator.rn.reactions[count]
            else:
                reactant_objects = [self.entries_list[i] for i in reaction[0]]

                product_objects = [self.entries_list[i] for i in reaction[1]]
                reaction_object = ConcertedReaction(reactant_objects, product_objects)

            self.add_reaction(reaction_object.graph_representation())
            count += 1
            if not self.add_concerteds:
                if count == len(reaction_iterator.rn.reactions):
                    break

        self.PR_record = self.build_PR_record()  # begin creating PR list
        self.Reactant_record = self.build_reactant_record()  # begin creating rct list

        print("init() end", time.time())

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

    @staticmethod
    def default_cost(free_energy: float) -> float:
        """
        Method to determine edge weight using exponent(dG/kt) + 1 cost function
        """
        return default_cost(free_energy)

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
            print(ii, len(new_solved_PRs), old_attrs != new_attrs, ii < max_iter)

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

    def find_path_cost(
        self,
        starts,
        weight,
        old_solved_PRs,
        cost_from_start,
        min_cost,
        PRs,
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
                    this_path = ReactionPath.characterize_path(
                        paths[node],
                        weight,
                        self.graph,
                    )
                    if node in this_path.all_prereqs:
                        wrong_paths[start].append(int(node))

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
                        path_dict_class = ReactionPath.characterize_path_final(
                            path,
                            self.weight,
                            self.graph,
                            self.solved_PRs,
                            self.PRs,
                            self.PR_byproducts,
                        )
                        heapq.heappush(
                            my_heapq, (path_dict_class.cost, next(c), path_dict_class)
                        )
        except Exception:
            print("no path from this start to the target", start)
        top_path_list = []
        while len(paths) < num_paths and my_heapq:
            (cost, _x, path_dict_class) = heapq.heappop(my_heapq)
            top_path_list.append(path_dict_class.path)
            print(
                len(paths),
                cost,
                path_dict_class.overall_free_energy_change,
                path_dict_class.hardest_step_deltaG,
                path_dict_class.path_dict,
            )
            paths.append(
                path_dict_class.path_dict
            )  # ideally just append the class, but for now dict for easy printing

        self.paths = paths
        self.top_path_list = top_path_list
        print("find_paths end", time.time())

        return self.PRs, paths, top_path_list


def path_finding_wrapper(
    mol_list: List[MoleculeEntry], init_mols: List[MoleculeEntry], target: MoleculeEntry
):
    entries_box = EntriesBox(mol_list)
    ri = ReactionIterator(entries_box)
    rn = ReactionNetwork(ri)

    initial_inds = [e.parameters["ind"] for e in init_mols]

    rn.solve_prerequisites(initial_inds, "default_cost")

    # set initial conditions
    PRs, paths, top_path_list = rn.find_paths(
        initial_inds, target.parameters["ind"], weight="default_cost", num_paths=20
    )
    # return shortest paths to every mol
    return PRs, paths, top_path_list


def reaction_string_to_dict(str, dG):
    split1 = str.split(",")
    reactants = split1[0].split("+")
    products = split1[1].split("+")
    return {"reactants": reactants, "products": products, "dG": dG}


def pathfinding_path_report(folder: str, rn: ReactionNetwork, paths):
    entries_dict = {}
    for entry in rn.entries_list:
        entries_dict[entry.parameters["ind"]] = entry

    if not os.path.isdir(folder):
        os.mkdir(folder)

    visualize_molecules(folder + "/molecule_diagrams", entries_dict)

    pathways = []
    for reaction_path in paths:
        pathway = []
        cost = reaction_path["cost"]
        for node in reaction_path["full_path"]:
            if type(node) == str:
                dG = rn.graph.nodes[node]["free_energy"]
                pathway.append(reaction_string_to_dict(node, dG))

        pathways.append((cost, pathway))

    with open(folder + "/pathway_report.tex", "w") as f:
        generate_latex_header(f)

        count = 1
        for cost, pathway in pathways:
            f.write("pathway " + str(count) + "\n\n")
            f.write("pathway cost: " + str(cost) + "\n\n")
            for reaction in pathway:
                latex_emit_reaction(f, reaction)

            f.write("\\newpage\n")
            count += 1

        generate_latex_footer(f)
