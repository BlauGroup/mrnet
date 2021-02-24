# coding: utf-8
import io
import os
import sys
import unittest
import copy
import pickle

from monty.serialization import dumpfn, loadfn
from networkx.readwrite import json_graph

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import RedoxReaction
from mrnet.network.reaction_network import ReactionPath, ReactionNetwork

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
        path = loadfn(os.path.join(test_dir, "unittest_characterize_path_path_IN.json"))

        solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_characterize_path_old_solved_PRs_IN.json")
        )

        with open(
            os.path.join(test_dir, "unittest_characterize_path_PRs_IN.pkl"), "rb"
        ) as input:
            PR_paths = pickle.load(input)

        # run calc
        path_instance = ReactionPath.characterize_path(
            path,
            "softplus",
            RN.min_cost,
            RN.graph,
            solved_PRs,
            RN.PR_byproducts,
            PR_paths,
        )

        # assert
        self.assertEqual(path_instance.byproducts, [356, 182, 548])
        self.assertEqual(path_instance.unsolved_prereqs, [])
        self.assertEqual(path_instance.solved_prereqs, [556, 46])
        self.assertEqual(path_instance.cost, 12.592087913497771)
        self.assertEqual(path_instance.pure_cost, 0.0)
        self.assertEqual(path_instance.hardest_step_deltaG, None)
        self.assertEqual(
            path_instance.path,
            [
                456,
                "456+PR_556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,356+543",
                543,
                "543+PR_46,15",
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
        path_class = ReactionPath.characterize_path_final(
            RN_pr_solved.PRs[2][456].path,
            RN_pr_solved.weight,
            RN_pr_solved.min_cost,
            RN_pr_solved.graph,
            RN_pr_solved.solved_PRs,
            RN_pr_solved.PR_byproducts,
            RN_pr_solved.PRs,
        )

        # assert
        self.assertEqual(path_class.byproducts, [356, 182, 548])
        self.assertEqual(path_class.solved_prereqs, [556, 46])
        self.assertEqual(path_class.all_prereqs, [556, 46])
        self.assertEqual(path_class.cost, 12.592087913497771)
        self.assertEqual(
            path_class.path,
            [
                456,
                "456+PR_556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,356+543",
                543,
                "543+PR_46,15",
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


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        if ob:
            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
            )
            cls.EC_mg = metal_edge_extender(EC_mg)

            LiEC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
            )
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            LEDC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LEDC.xyz")), OpenBabelNN()
            )
            cls.LEDC_mg = metal_edge_extender(LEDC_mg)

            LEMC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")), OpenBabelNN()
            )
            cls.LEMC_mg = metal_edge_extender(LEMC_mg)

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

            # dumpfn(cls.LiEC_reextended_entries, "unittest_input_molentries.json")

            with open(os.path.join(test_dir, "unittest_RN_build.pkl"), "rb") as input:
                cls.RN_build = pickle.load(input)

            with open(
                os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb"
            ) as input:
                cls.RN_pr_solved = pickle.load(input)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_add_reactions(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(
            self.LiEC_reextended_entries, electron_free_energy=-2.15
        )

        # set up input variables
        EC_0_entry = None
        EC_minus_entry = None

        # print(RN.entries["C3 H4 O3"].keys())

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_0_entry = entry
                break
        for entry in RN.entries["C3 H4 O3"][10][-1]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_minus_entry = entry
                break

        redox = RedoxReaction(EC_0_entry, EC_minus_entry)
        redox.electron_free_energy = -2.15
        redox_graph = redox.graph_representation()

        # run calc
        RN.add_reaction(redox_graph)

        # assert
        self.assertEqual(list(RN.graph.nodes), ["456,455", 456, 455, "455,456"])
        self.assertEqual(
            list(RN.graph.edges),
            [("456,455", 455), (456, "456,455"), (455, "455,456"), ("455,456", 456)],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(
            self.LiEC_reextended_entries, electron_free_energy=-2.15
        )

        # perfrom calc
        RN.build()

        # assert
        EC_ind = None
        LEDC_ind = None
        LiEC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        self.assertEqual(len(RN.entries_list), 569)
        self.assertEqual(EC_ind, 456)
        self.assertEqual(LEDC_ind, 511)
        self.assertEqual(Li1_ind, 556)
        self.assertEqual(LiEC_ind, 424)

        self.assertEqual(len(RN.graph.nodes), 10481)
        self.assertEqual(len(RN.graph.edges), 22890)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build_PR_record(self):
        # set up RN
        RN = copy.deepcopy(self.RN_build)

        # run calc
        PR_record = RN.build_PR_record()

        # assert
        self.assertEqual(len(PR_record[0]), 42)
        self.assertEqual(PR_record[44], ["165+PR_44,434"])
        self.assertEqual(len(PR_record[529]), 0)
        self.assertEqual(len(PR_record[556]), 104)
        self.assertEqual(len(PR_record[564]), 165)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build_reactant_record(self):

        # set up RN
        RN = copy.deepcopy(self.RN_build)

        # run calc
        reactant_record = RN.build_reactant_record()

        # assert
        self.assertEqual(len(reactant_record[0]), 43)
        self.assertCountEqual(
            reactant_record[44], ["44+PR_165,434", "44,43", "44,40+556"]
        )
        self.assertEqual(len(reactant_record[529]), 0)
        self.assertEqual(len(reactant_record[556]), 104)
        self.assertEqual(len(reactant_record[564]), 167)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_solve_prerequisites(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # set up RN
        RN = copy.deepcopy(self.RN_build)
        RN.build_PR_record()
        # set up input variables

        EC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        # perfrom calc
        PRs_calc, old_solved_PRs = RN.solve_prerequisites(
            [EC_ind, Li1_ind], weight="softplus"
        )

        # assert
        PR_paths = RN_pr_solved.PRs

        for node in PRs_calc:
            for start in PRs_calc[node]:
                self.assertEqual(
                    [
                        PRs_calc[node][start].all_prereqs,
                        PRs_calc[node][start].byproducts,
                        PRs_calc[node][start].full_path,
                        PRs_calc[node][start].path,
                        PRs_calc[node][start].solved_prereqs,
                        PRs_calc[node][start].unsolved_prereqs,
                    ],
                    [
                        PR_paths[node][start].all_prereqs,
                        PR_paths[node][start].byproducts,
                        PR_paths[node][start].full_path,
                        PR_paths[node][start].path,
                        PR_paths[node][start].solved_prereqs,
                        PR_paths[node][start].unsolved_prereqs,
                    ],
                )

                if PRs_calc[node][start].cost != PR_paths[node][start].cost:
                    self.assertAlmostEqual(
                        PRs_calc[node][start].cost, PR_paths[node][start].cost, places=2
                    )
                if PRs_calc[node][start].pure_cost != PR_paths[node][start].pure_cost:
                    self.assertAlmostEqual(
                        PRs_calc[node][start].pure_cost,
                        PR_paths[node][start].pure_cost,
                        places=2,
                    )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_path_cost(self):

        # set up RN

        with open(
            os.path.join(test_dir, "unittest_RN_pr_ii_4_before_find_path_cost.pkl"),
            "rb",
        ) as input:
            RN_pr_ii_4 = pickle.load(input)

        # set up input variables
        EC_ind = None
        for entry in RN_pr_ii_4.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN_pr_ii_4.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN_pr_ii_4.entries["Li1"][0][1][0].parameters["ind"]

        loaded_cost_from_start_str = loadfn(
            os.path.join(test_dir, "unittest_find_path_cost_cost_from_start_IN.json")
        )

        old_solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_find_path_cost_old_solved_prs_IN.json")
        )

        loaded_min_cost_str = loadfn(
            os.path.join(test_dir, "unittest_find_path_cost_min_cost_IN.json")
        )

        with open(
            os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"), "rb"
        ) as input:
            loaded_PRs = pickle.load(input)

        loaded_cost_from_start = {}
        for node in loaded_cost_from_start_str:
            loaded_cost_from_start[int(node)] = {}
            for start in loaded_cost_from_start_str[node]:
                loaded_cost_from_start[int(node)][
                    int(start)
                ] = loaded_cost_from_start_str[node][start]

        loaded_min_cost = {}
        for node in loaded_min_cost_str:
            loaded_min_cost[int(node)] = loaded_min_cost_str[node]

        # perform calc
        PRs_cal, cost_from_start_cal, min_cost_cal = RN_pr_ii_4.find_path_cost(
            [EC_ind, Li1_ind],
            RN_pr_ii_4.weight,
            old_solved_PRs,
            loaded_cost_from_start,
            loaded_min_cost,
            loaded_PRs,
        )

        # assert
        self.assertEqual(cost_from_start_cal[456][456], 0.0)
        self.assertEqual(cost_from_start_cal[556][456], "no_path")
        self.assertEqual(cost_from_start_cal[0][456], 7.291618376702763)
        self.assertEqual(cost_from_start_cal[6][556], 3.76319246637113)
        self.assertEqual(cost_from_start_cal[80][456], 9.704537989094758)

        self.assertEqual(min_cost_cal[556], 0.0)
        self.assertEqual(min_cost_cal[1], 8.01719987083261)
        self.assertEqual(min_cost_cal[4], 5.8903562531872256)
        self.assertEqual(min_cost_cal[148], 2.5722550049918964)

        self.assertEqual(PRs_cal[556][556].path, [556])
        self.assertEqual(PRs_cal[556][456].path, None)
        self.assertEqual(PRs_cal[29][456].path, None)
        self.assertEqual(PRs_cal[313], {})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_identify_solved_PRs(self):

        # set up RN
        with open(
            os.path.join(
                test_dir, "unittest_RN_pr_ii_4_before_identify_solved_PRs.pkl"
            ),
            "rb",
        ) as input:
            RN_pr_ii_4 = pickle.load(input)

        # set up input variables
        cost_from_start_IN_str = loadfn(
            os.path.join(
                test_dir, "unittest_identify_solved_PRs_cost_from_start_IN.json"
            )
        )
        solved_PRs = loadfn(
            os.path.join(test_dir, "unittest_identify_solved_PRs_solved_PRs_IN.json")
        )
        with open(
            os.path.join(test_dir, "unittest_find_path_cost_PRs_IN.pkl"), "rb"
        ) as input:
            PRs = pickle.load(input)

        cost_from_start = {}
        for node in cost_from_start_IN_str:
            cost_from_start[int(node)] = {}
            for start in cost_from_start_IN_str[node]:
                cost_from_start[int(node)][int(start)] = cost_from_start_IN_str[node][
                    start
                ]

        # perform calc
        (
            solved_PRs_cal,
            new_solved_PRs_cal,
            cost_from_start_cal,
        ) = RN_pr_ii_4.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

        # assert
        self.assertEqual(len(solved_PRs_cal), 105)
        self.assertEqual(len(cost_from_start_cal), 569)
        self.assertEqual(cost_from_start_cal[456][556], "no_path")
        self.assertEqual(cost_from_start_cal[556][556], 0.0)
        self.assertEqual(cost_from_start_cal[2][556], 9.902847048351545)
        self.assertEqual(cost_from_start_cal[7][456], 3.5812151003313524)
        self.assertEqual(cost_from_start_cal[30][556], "no_path")

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_update_edge_weights(self):

        # set up RN

        with open(
            os.path.join(
                test_dir, "unittest_RN_pr_ii_4_before_update_edge_weights.pkl"
            ),
            "rb",
        ) as input:
            RN_pr_ii_4 = pickle.load(input)

        # set up input variables
        min_cost_str = loadfn(
            os.path.join(test_dir, "unittest_update_edge_weights_min_cost_IN.json")
        )
        with open(
            os.path.join(test_dir, "unittest_update_edge_weights_orig_graph_IN.pkl"),
            "rb",
        ) as input:
            orig_graph = pickle.load(input)

        min_cost = {}
        for key in min_cost_str:
            min_cost[int(key)] = min_cost_str[key]

        # perform calc
        attrs_cal = RN_pr_ii_4.update_edge_weights(min_cost, orig_graph)

        # assert
        self.assertEqual(len(attrs_cal), 6143)
        self.assertEqual(
            attrs_cal[(556, "556+PR_456,421")]["softplus"], 0.24363920804933614
        )
        self.assertEqual(
            attrs_cal[(41, "41+PR_556,42")]["softplus"], 0.26065563056500646
        )
        self.assertEqual(
            attrs_cal[(308, "308+PR_556,277")]["softplus"], 0.08666835894406484
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_final_PR_check(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # set up RN
        RN = copy.deepcopy(RN_pr_solved)

        # perform calc
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        RN.final_PR_check(RN.PRs)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout

        # assert
        self.assertTrue(output.__contains__("No path found from any start to PR 30"))
        self.assertTrue(
            output.__contains__("WARNING: Matching prereq and byproduct found! 46")
        )
        self.assertTrue(output.__contains__("No path found from any start to PR 513"))
        self.assertTrue(output.__contains__("No path found from any start to PR 539"))

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_or_remove_bad_nodes(self):

        # set up RN
        RN = copy.deepcopy(self.RN_build)

        # set up input variables
        LEDC_ind = None
        LiEC_ind = None
        EC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break

        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break

        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break

        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        nodes = [LEDC_ind, LiEC_ind, Li1_ind, EC_ind]

        # perform calc & assert
        bad_nodes_list = RN.find_or_remove_bad_nodes(nodes, remove_nodes=False)
        self.assertEqual(len(bad_nodes_list), 231)
        self.assertTrue(
            {"511,108+112", "46+PR_556,34", "556+PR_199,192", "456,399+543", "456,455"}
            <= set(bad_nodes_list)
        )

        bad_nodes_pruned_graph = RN.find_or_remove_bad_nodes(nodes, remove_nodes=True)
        self.assertEqual(len(bad_nodes_pruned_graph.nodes), 10254)
        self.assertEqual(len(bad_nodes_pruned_graph.edges), 22424)
        for node_ind in nodes:
            self.assertEqual(bad_nodes_pruned_graph[node_ind], {})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_valid_shortest_simple_paths(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        RN = copy.deepcopy(RN_pr_solved)

        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break

        paths = RN.valid_shortest_simple_paths(EC_ind, LEDC_ind)
        p = [
            [
                456,
                "456+PR_556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420+PR_41,511",
                511,
            ],
            [
                456,
                "456+PR_556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+PR_420,511",
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
                "51+PR_556,41",
                41,
                "41+PR_420,511",
                511,
            ],
            [
                456,
                "456+PR_556,421",
                421,
                "421,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420+PR_41,511",
                511,
            ],
            [
                456,
                "456+PR_556,421",
                421,
                "421,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+PR_420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448+PR_556,420",
                420,
                "420,41+164",
                41,
                "41+PR_420,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455,448",
                448,
                "448+PR_556,420",
                420,
                "420+PR_41,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455+PR_556,423",
                423,
                "423,420",
                420,
                "420+PR_41,511",
                511,
            ],
            [
                456,
                "456,455",
                455,
                "455+PR_556,423",
                423,
                "423,420",
                420,
                "420,41+164",
                41,
                "41+PR_420,511",
                511,
            ],
            [
                456,
                "456+PR_556,424",
                424,
                "424,423",
                423,
                "423,420",
                420,
                "420,419",
                419,
                "419+PR_41,510",
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
        self.assertCountEqual(p, paths_generated)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_paths(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        # set up RN
        RN = copy.deepcopy(RN_pr_solved)

        # set up input variables
        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        PR_paths_calculated, paths_calculated, top_paths_list = RN.find_paths(
            [EC_ind, Li1_ind], LEDC_ind, weight="softplus", num_paths=10
        )

        if 420 in paths_calculated[0]["all_prereqs"]:
            self.assertEqual(paths_calculated[0]["byproducts"], [164])
        elif 41 in paths_calculated[0]["all_prereqs"]:
            self.assertEqual(paths_calculated[0]["byproducts"], [164, 164])

        self.assertAlmostEqual(paths_calculated[0]["cost"], 2.3135953094636403, 5)
        self.assertAlmostEqual(
            paths_calculated[0]["overall_free_energy_change"], -6.2399175587598394, 5
        )
        self.assertAlmostEqual(
            paths_calculated[0]["hardest_step_deltaG"], 0.37075842588456, 5
        )

        for path in paths_calculated:
            self.assertTrue(abs(path["cost"] - path["pure_cost"]) < 0.000000001)

    def test_mols_w_cuttoff(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        RN_loaded = copy.deepcopy(RN_pr_solved)

        mols_to_keep, pruned_entries_list = ReactionNetwork.mols_w_cuttoff(
            RN_loaded, 0, build_pruned_network=False
        )

        self.assertEqual(len(mols_to_keep), 196)

    def test_identify_concerted_rxns_via_intermediates(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        RN_loaded = copy.deepcopy(RN_pr_solved)

        with open(
            os.path.join(test_dir, "RN_unittest_pruned_mols_to_keep.json"), "rb"
        ) as handle:
            mols_to_keep = pickle.load(handle)

        reactions = ReactionNetwork.identify_concerted_rxns_via_intermediates(
            RN_loaded, mols_to_keep, single_elem_interm_ignore=["C1", "H1", "O1", "Li1"]
        )

        self.assertEqual(len(reactions), 2410)

    def test_add_concerted_rxns(self):
        with open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "rb") as input:
            RN_pr_solved = pickle.load(input)

        RN_loaded = copy.deepcopy(RN_pr_solved)

        with open(
            os.path.join(test_dir, "RN_unittest_reactions_list.json"), "rb"
        ) as handle:
            reactions = pickle.load(handle)

        RN_loaded.add_concerted_rxns(RN_loaded, RN_loaded, reactions)

        self.assertEqual(len(RN_loaded.graph.nodes), 15064)
        self.assertEqual(len(RN_loaded.graph.edges), 36589)


if __name__ == "__main__":
    unittest.main()
