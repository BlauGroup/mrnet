import copy
from typing import Dict, List, Tuple, Union, Any, FrozenSet, Set, TypeVar
import networkx as nx
from monty.json import MSONable

from mrnet.core.reactions import (
    exponent,
    rexp,
    softplus,
    default_cost,
)
from mrnet.utils.classes import load_class

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


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
                        class_instance.pure_cost += softplus(
                            graph.nodes[step]["free_energy"]
                        )
                    elif weight == "exponent":
                        class_instance.pure_cost += exponent(
                            graph.nodes[step]["free_energy"]
                        )
                    elif weight == "rexp":
                        class_instance.pure_cost += rexp(
                            graph.nodes[step]["free_energy"]
                        )
                    elif weight == "default_cost":
                        class_instance.pure_cost += default_cost(
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
