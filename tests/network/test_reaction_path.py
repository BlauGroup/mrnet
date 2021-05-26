# coding: utf-8
import io
import os
import sys
import unittest
import copy
import pickle
from itertools import permutations
from ast import literal_eval

from monty.serialization import dumpfn, loadfn
from networkx.readwrite import json_graph

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import RedoxReaction
from mrnet.network.reaction_network import ReactionPath, ReactionNetwork
from mrnet.network.reaction_generation import ReactionGenerator

try:
    import openbabel as ob
except ImportError:
    ob = None

test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


class TestReactionPath(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_characterize_path(self):
        # set up input variables
        with open(
            os.path.join(test_dir, "unittest_RN_before_characterize_path.pkl"), "rb"
        ) as input:
            RN = pickle.load(input)

        solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_characterize_path_old_solved_PRs_IN.json")
        )
        path = loadfn(os.path.join(test_dir, "unittest_characterize_path_path_IN.json"))

        # run calc
        path_instance = ReactionPath.characterize_path(
            path, "softplus", RN.graph, solved_PRs
        )

        # assert
        self.assertEqual(path_instance.byproducts, [356, 548])
        self.assertEqual(path_instance.unsolved_prereqs, [])
        self.assertEqual(path_instance.solved_prereqs, [556, 46])
        self.assertEqual(path_instance.cost, 12.592087913497771)
        self.assertEqual(path_instance.pure_cost, 0.0)
        self.assertEqual(path_instance.hardest_step_deltaG, None)
        self.assertEqual(
            path_instance.path,
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,356+543",
                543,
                "46+543,15",
                15,
                "15,13",
                13,
                "13,1+548",
                1,
                "1,2",
                2,
            ],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_characterize_path_final(self):

        # set up input variables
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # perform calc
        # print(RN_pr_solved.PRs[2], RN_pr_solved.PRs[2])
        path_class = ReactionPath.characterize_path_final(
            RN_pr_solved.PRs[2][456].path,
            RN_pr_solved.weight,
            RN_pr_solved.graph,
            RN_pr_solved.solved_PRs,
            RN_pr_solved.PRs,
            RN_pr_solved.PR_byproducts,
        )

        # assert
        self.assertEqual(path_class.byproducts, [356, 548, 182])
        self.assertEqual(path_class.solved_prereqs, [556, 46])
        self.assertEqual(path_class.all_prereqs, [556, 46])
        self.assertEqual(path_class.cost, 12.592087913497771)
        self.assertEqual(
            path_class.path,
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,356+543",
                543,
                "46+543,15",
                15,
                "15,13",
                13,
                "13,1+548",
                1,
                "1,2",
                2,
            ],
        )
        self.assertEqual(path_class.overall_free_energy_change, 10.868929712195717)
        self.assertEqual(path_class.pure_cost, 12.592087913497771)
        self.assertEqual(path_class.hardest_step_deltaG, 4.018404627691188)


if __name__ == "__main__":
    unittest.main()
