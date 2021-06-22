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
from mrnet.network.reaction_generation import (
    ReactionGenerator,
    EntriesBox,
    ReactionIterator,
)

import openbabel as ob


test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        EC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
        )
        EC_mg = metal_edge_extender(EC_mg)

        LiEC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
        )
        LiEC_mg = metal_edge_extender(LiEC_mg)

        LEDC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LEDC.xyz")), OpenBabelNN()
        )
        LEDC_mg = metal_edge_extender(LEDC_mg)

        LEMC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")), OpenBabelNN()
        )
        LEMC_mg = metal_edge_extender(LEMC_mg)

        cls.LiEC_reextended_entries = []
        entries = loadfn(os.path.join(test_dir, "LiEC_reextended_entries.json"))
        for entry in entries:
            if "optimized_molecule" in entry["output"]:
                mol = entry["output"]["optimized_molecule"]
            else:
                mol = entry["output"]["initial_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(
                molecule=mol,
                energy=E,
                enthalpy=H,
                entropy=S,
                entry_id=entry["task_id"],
            )
            if mol_entry.formula == "Li1":
                if mol_entry.charge == 1:
                    cls.LiEC_reextended_entries.append(mol_entry)
            else:
                cls.LiEC_reextended_entries.append(mol_entry)

        cls.entries_box = EntriesBox(
            cls.LiEC_reextended_entries, remove_complexes=False
        )

        cls.RI = ReactionIterator(cls.entries_box)
        cls.RN = ReactionNetwork(cls.RI, add_concerteds=False)

        # set up input variables
        cls.LEDC_ind = None
        cls.LiEC_ind = None
        cls.EC_ind = None

        for entry in cls.entries_box.entries_dict["C3 H4 O3"][10][0]:
            if EC_mg.isomorphic_to(entry.mol_graph):
                cls.EC_ind = entry.parameters["ind"]
                break

        for entry in cls.entries_box.entries_dict["C4 H4 Li2 O6"][17][0]:
            if LEDC_mg.isomorphic_to(entry.mol_graph):
                cls.LEDC_ind = entry.parameters["ind"]
                break

        for entry in cls.entries_box.entries_dict["C3 H4 Li1 O3"][12][1]:
            if LiEC_mg.isomorphic_to(entry.mol_graph):
                cls.LiEC_ind = entry.parameters["ind"]
                break

        cls.Li1_ind = cls.entries_box.entries_dict["Li1"][0][1][0].parameters["ind"]

        print("LEDC_ind:", cls.LEDC_ind)
        print("LiEC_ind:", cls.LiEC_ind)
        print("EC_ind:", cls.EC_ind)
        print("Li1_ind:", cls.Li1_ind)

        cls.RN_solved = copy.deepcopy(cls.RN)
        cls.RN_solved.solve_prerequisites([cls.EC_ind, cls.Li1_ind], weight="softplus")

        # dumpfn(cls.LiEC_reextended_entries, "unittest_input_molentries.json")

        # with open(os.path.join(test_dir, "unittest_RN_build.pkl"), "rb") as input:
        #     cls.RN_build = pickle.load(input)

        # with open(
        #     os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb"
        # ) as input:
        #     cls.RN_pr_solved = pickle.load(input)

    def test_build_PR_record(self):
        PR_record = self.RN.build_PR_record()
        assert len(PR_record[0]) == 42
        assert PR_record[44] == [(165, "44+165,434")]
        assert len(PR_record[529]) == 0
        assert len(PR_record[self.Li1_ind]) == 104
        assert len(PR_record[564]) == 165

    def test_build_reactant_record(self):
        reactant_record = self.RN.build_reactant_record()
        assert len(reactant_record[0]) == 43
        assert len(reactant_record[44]) == 3
        assert set(reactant_record[44]) == set(
            [(44, "44+165,434"), (44, "44,43"), (44, "44,40+556")]
        )
        assert len(reactant_record[529]) == 0
        assert len(reactant_record[self.Li1_ind]) == 104
        assert len(reactant_record[564]) == 167

    # @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    # def test_solve_prerequisites(self):
    #     # set up RN
    #     RN = copy.deepcopy(self.RN_build)
    #     RN.build_PR_record()
    #     # set up input variables

    #     EC_ind = None

    #     for entry in RN.entries["C3 H4 O3"][10][0]:
    #         if self.EC_mg.isomorphic_to(entry.mol_graph):
    #             EC_ind = entry.parameters["ind"]
    #         break
    #     Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

    #     # perfrom calc
    #     PRs_calc, old_solved_PRs = RN.solve_prerequisites(
    #         [EC_ind, Li1_ind], weight="softplus"
    #     )

    #     # assert
    #     with open(
    #         os.path.join(test_dir, "unittest_RN_pr_solved_PRs.pkl"), "rb"
    #     ) as input:
    #         PR_paths = pickle.load(input)
    #     print(PRs_calc[12])
    #     print(PR_paths[12])

    #     for node in PR_paths:
    #         for start in PR_paths[node]:
    #             try:
    #                 self.assertEqual(
    #                     PRs_calc[node][start].all_prereqs,
    #                     PR_paths[node][start].all_prereqs,
    #                 )
    #                 self.assertEqual(
    #                     PRs_calc[node][start].byproducts,
    #                     PR_paths[node][start].byproducts,
    #                 )
    #                 self.assertEqual(
    #                     PRs_calc[node][start].solved_prereqs,
    #                     PR_paths[node][start].solved_prereqs,
    #                 )
    #                 self.assertEqual(
    #                     PRs_calc[node][start].unsolved_prereqs,
    #                     PR_paths[node][start].unsolved_prereqs,
    #                 )
    #                 # below code (similar to many other functions, is responsible for tweaking old strings to match current representation)
    #                 for i in range(
    #                     len(PR_paths[node][start].full_path)
    #                     if PR_paths[node][start].full_path != None
    #                     else 0
    #                 ):  # iterate over all nodes in the path
    #                     path = PR_paths[node][start].full_path[i]
    #                     if isinstance(
    #                         path, str
    #                     ):  # for string nodes, remove PR from the node name
    #                         trimmed_pr = path.replace("PR_", "")
    #                     else:
    #                         trimmed_pr = path
    #                     # order of reactants no longer enforced by PR, so just make sure all reactants are present in the reaction rather than worrying about exact order
    #                     try:
    #                         self.assertEqual(
    #                             trimmed_pr, PRs_calc[node][start].full_path[i]
    #                         )
    #                     except AssertionError:
    #                         rct_path = trimmed_pr.split(",")[0].split("+")
    #                         rct_calc = (
    #                             PRs_calc[node][start]
    #                             .full_path[i]
    #                             .split(",")[0]
    #                             .split("+")
    #                         )
    #                         self.assertCountEqual(rct_path, rct_calc)
    #                 for i in range(
    #                     len(PR_paths[node][start].path)
    #                     if PR_paths[node][start].path != None
    #                     else 0
    #                 ):
    #                     path = PR_paths[node][start].path[i]
    #                     if isinstance(path, str):
    #                         trimmed_pr = path.replace("PR_", "")
    #                     else:
    #                         trimmed_pr = path
    #                     try:
    #                         self.assertEqual(trimmed_pr, PRs_calc[node][start].path[i])
    #                     except AssertionError:
    #                         rct_path = trimmed_pr.split(",")[0].split("+")
    #                         rct_calc = (
    #                             PRs_calc[node][start].path[i].split(",")[0].split("+")
    #                         )
    #                         self.assertCountEqual(rct_path, rct_calc)

    #                 if PRs_calc[node][start].cost != PR_paths[node][start].cost:
    #                     self.assertAlmostEqual(
    #                         PRs_calc[node][start].cost,
    #                         PR_paths[node][start].cost,
    #                         places=2,
    #                     )
    #                 if (
    #                     PRs_calc[node][start].pure_cost
    #                     != PR_paths[node][start].pure_cost
    #                 ):
    #                     self.assertAlmostEqual(
    #                         PRs_calc[node][start].pure_cost,
    #                         PR_paths[node][start].pure_cost,
    #                         places=2,
    #                     )
    #             except KeyError:
    #                 self.assertTrue(False)
    #                 print("Node: ", node)
    #                 print("Start: ", start)

    # @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    # def test_find_path_cost(self):
    #     # set up RN

    #     with open(
    #         os.path.join(test_dir, "unittest_RN_pr_ii_4_before_find_path_cost.pkl"),
    #         "rb",
    #     ) as input:
    #         RN_pr_ii_4 = pickle.load(input)

    #     # set up input variables
    #     EC_ind = None
    #     for entry in RN_pr_ii_4.entries["C3 H4 O3"][10][0]:
    #         if self.EC_mg.isomorphic_to(entry.mol_graph):
    #             EC_ind = entry.parameters["ind"]
    #             break
    #     for entry in RN_pr_ii_4.entries["C4 H4 Li2 O6"][17][0]:
    #         if self.LEDC_mg.isomorphic_to(entry.mol_graph):
    #             LEDC_ind = entry.parameters["ind"]
    #             break
    #     Li1_ind = RN_pr_ii_4.entries["Li1"][0][1][0].parameters["ind"]

    #     loaded_cost_from_start_str = loadfn(
    #         os.path.join(test_dir, "unittest_find_path_cost_cost_from_start_IN.json")
    #     )

    #     old_solved_PRs = loadfn(
    #         os.path.join(test_dir, "unittest_find_path_cost_old_solved_prs_IN.json")
    #     )

    #     loaded_min_cost_str = loadfn(
    #         os.path.join(test_dir, "unittest_find_path_cost_min_cost_IN.json")
    #     )

    #     with open(
    #         os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"), "rb"
    #     ) as input:
    #         loaded_PRs = pickle.load(input)

    #     loaded_cost_from_start = {}
    #     for node in loaded_cost_from_start_str:
    #         loaded_cost_from_start[int(node)] = {}
    #         for start in loaded_cost_from_start_str[node]:
    #             loaded_cost_from_start[int(node)][
    #                 int(start)
    #             ] = loaded_cost_from_start_str[node][start]

    #     loaded_min_cost = {}
    #     for node in loaded_min_cost_str:
    #         loaded_min_cost[int(node)] = loaded_min_cost_str[node]

    #     # perform calc
    #     PRs_cal, cost_from_start_cal, min_cost_cal = RN_pr_ii_4.find_path_cost(
    #         [EC_ind, Li1_ind],
    #         RN_pr_ii_4.weight,
    #         old_solved_PRs,
    #         loaded_cost_from_start,
    #         loaded_min_cost,
    #         loaded_PRs,
    #     )

    #     # assert
    #     self.assertEqual(cost_from_start_cal[456][456], 0.0)
    #     self.assertEqual(cost_from_start_cal[556][456], "no_path")
    #     self.assertEqual(cost_from_start_cal[0][456], 7.291618376702763)
    #     self.assertEqual(cost_from_start_cal[6][556], 3.76319246637113)
    #     self.assertEqual(cost_from_start_cal[80][456], 9.704537989094758)

    #     self.assertEqual(min_cost_cal[556], 0.0)
    #     self.assertEqual(min_cost_cal[1], 8.01719987083261)
    #     self.assertEqual(min_cost_cal[4], 5.8903562531872256)
    #     self.assertEqual(min_cost_cal[148], 2.5722550049918964)

    #     self.assertEqual(PRs_cal[556][556].path, [556])
    #     self.assertEqual(PRs_cal[556][456].path, None)
    #     self.assertEqual(PRs_cal[29][456].path, None)
    #     self.assertEqual(PRs_cal[313], {})

    # @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    # def test_identify_solved_PRs(self):

    #     # set up RN
    #     with open(
    #         os.path.join(
    #             test_dir, "unittest_RN_pr_ii_4_before_identify_solved_PRs.pkl"
    #         ),
    #         "rb",
    #     ) as input:
    #         RN_pr_ii_4 = pickle.load(input)

    #     # set up input variables
    #     cost_from_start_IN_str = loadfn(
    #         os.path.join(
    #             test_dir, "unittest_identify_solved_PRs_cost_from_start_IN.json"
    #         )
    #     )
    #     solved_PRs = loadfn(
    #         os.path.join(test_dir, "unittest_identify_solved_PRs_solved_PRs_IN.json")
    #     )
    #     with open(
    #         os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"), "rb"
    #     ) as input:
    #         PRs = pickle.load(input)

    #     cost_from_start = {}
    #     for node in cost_from_start_IN_str:
    #         cost_from_start[int(node)] = {}
    #         for start in cost_from_start_IN_str[node]:
    #             cost_from_start[int(node)][int(start)] = cost_from_start_IN_str[node][
    #                 start
    #             ]

    #     # perform calc
    #     (
    #         solved_PRs_cal,
    #         new_solved_PRs_cal,
    #         cost_from_start_cal,
    #     ) = RN_pr_ii_4.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

    #     # assert
    #     self.assertEqual(len(solved_PRs_cal), 166)
    #     self.assertEqual(len(cost_from_start_cal), 569)
    #     self.assertEqual(cost_from_start_cal[456][556], "no_path")
    #     self.assertEqual(cost_from_start_cal[556][556], 0.0)
    #     self.assertEqual(cost_from_start_cal[2][556], 9.902847048351545)
    #     self.assertEqual(cost_from_start_cal[7][456], 3.5812151003313524)
    #     self.assertEqual(cost_from_start_cal[30][556], "no_path")

    # @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    # def test_update_edge_weights(self):

    #     # set up RN

    #     with open(
    #         os.path.join(
    #             test_dir, "unittest_RN_pr_ii_4_before_update_edge_weights.pkl"
    #         ),
    #         "rb",
    #     ) as input:
    #         RN_pr_ii_4 = pickle.load(input)

    #     # set up input variables
    #     min_cost_str = loadfn(
    #         os.path.join(test_dir, "unittest_update_edge_weights_min_cost_IN.json")
    #     )
    #     with open(
    #         os.path.join(test_dir, "unittest_update_edge_weights_orig_graph_IN.pkl"),
    #         "rb",
    #     ) as input:
    #         orig_graph = pickle.load(input)

    #     min_cost = {}
    #     for key in min_cost_str:
    #         temp = min_cost_str[key]
    #         min_cost[int(key)] = temp

    #     # perform calc
    #     attrs_cal = RN_pr_ii_4.update_edge_weights(min_cost, orig_graph)

    #     # assert
    #     self.assertEqual(len(attrs_cal), 6143)
    #     self.assertEqual(
    #         attrs_cal[(556, "456+556,421")]["softplus"], 0.24363920804933614
    #     )
    #     self.assertEqual(attrs_cal[(41, "41+556,42")]["softplus"], 0.26065563056500646)
    #     self.assertEqual(
    #         attrs_cal[(308, "308+556,277")]["softplus"], 0.08666835894406484
    #     )

    # @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    # def test_final_PR_check(self):
    #     with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
    #         RN_pr_solved = pickle.load(input)

    #     # set up RN
    #     RN = copy.deepcopy(RN_pr_solved)

    #     # perform calc
    #     old_stdout = sys.stdout
    #     new_stdout = io.StringIO()
    #     sys.stdout = new_stdout
    #     RN.final_PR_check(RN.PRs)
    #     output = new_stdout.getvalue()
    #     sys.stdout = old_stdout

    #     # assert
    #     self.assertTrue(output.__contains__("No path found from any start to PR 30"))
    #     self.assertTrue(
    #         output.__contains__("NOTE: Matching prereq and byproduct found! 542")
    #     )
    #     self.assertTrue(output.__contains__("No path found from any start to PR 513"))
    #     self.assertTrue(output.__contains__("No path found from any start to PR 539"))

    def test_find_or_remove_bad_nodes(self):
        # set up RN
        RN = copy.deepcopy(self.RN)

        nodes = [self.LEDC_ind, self.LiEC_ind, self.Li1_ind, self.EC_ind]

        # perform calc & assert
        bad_nodes_list = RN.find_or_remove_bad_nodes(nodes, remove_nodes=False)
        assert len(bad_nodes_list) == 231
        assert {
            "511,108+112",
            "46+556,34",
            "199+556,192",
            "456,399+543",
            "456,455",
        } <= set(bad_nodes_list)

        bad_nodes_pruned_graph = RN.find_or_remove_bad_nodes(nodes, remove_nodes=True)
        # self.assertEqual(len(bad_nodes_pruned_graph.nodes), 10254)
        # self.assertEqual(len(bad_nodes_pruned_graph.edges), 22424)
        for node_ind in nodes:
            assert bad_nodes_pruned_graph[node_ind] == {}

    def test_valid_shortest_simple_paths(self):
        paths = self.RN_solved.valid_shortest_simple_paths(self.EC_ind, self.LEDC_ind)
        p = [
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448,51+164",
                51,
                "51+556,41",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,421",
                421,
                "421,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,421",
                421,
                "421,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448+556,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448+556,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455+556,423",
                423,
                "423,420",
                420,
                "41+420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455+556,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+420,511",
                511,
            ],
            [
                456,
                "456+556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,419",
                419,
                "41+419,510",
                510,
                "510,511",
                511,
            ],
        ]
        ind = 0
        paths_generated = []
        for path in paths:
            if ind == 10:
                break
            else:
                paths_generated.append(path)
                ind += 1
        for path in p:
            assert p.count(path) == paths_generated.count(path)

    def test_find_paths(self):
        (
            PR_paths_calculated,
            paths_calculated,
            top_paths_list,
        ) = self.RN_solved.find_paths(
            [self.EC_ind, self.Li1_ind], self.LEDC_ind, weight="softplus", num_paths=10
        )

        assert paths_calculated[0]["byproducts"] == [164]

        benchmark = 0.000001
        assert abs(paths_calculated[0]["cost"] - 2.3135953094636403) < benchmark
        assert (
            abs(paths_calculated[0]["overall_free_energy_change"] - -6.2399175587598394)
            < benchmark
        )
        assert abs(
            paths_calculated[0]["hardest_step_deltaG"] - 0.37075842588456 < benchmark
        )

        for path in paths_calculated:
            assert abs(path["cost"] - path["pure_cost"]) < 0.000000001


if __name__ == "__main__":
    unittest.main()
