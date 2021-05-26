import subprocess
import unittest
import copy
import pickle
import math
import os

import numpy as np
from scipy.constants import N_A
from monty.serialization import loadfn, dumpfn
from pymatgen.util.testing import PymatgenTest

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.network.reaction_generation import (
    ReactionIterator,
    EntriesBox,
    ReactionGenerator,
)
from mrnet.stochastic.serialize import (
    SerializeNetwork,
    serialize_simulation_parameters,
    find_mol_entry_from_xyz_and_charge,
    run_simulator,
    clone_database,
    serialize_initial_state,
)
from mrnet.stochastic.analyze import SimulationAnalyzer, NetworkUpdater
from mrnet.utils.constants import ROOM_TEMP

import openbabel as ob

__author__ = "Daniel Barter, Sam Blau"

root_test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
)

test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


class TestEntriesBox(PymatgenTest):
    def test_filter(self):
        molecule_entries = loadfn(
            os.path.join(root_test_dir, "choli_limited_complex_filter.json")
        )
        entries_box = EntriesBox(molecule_entries)
        assert len(entries_box.entries_list) == 100
        entries_unfiltered = EntriesBox(molecule_entries, remove_complexes=False)
        assert len(entries_unfiltered.entries_list) == 200


class TestReactionIterator(PymatgenTest):
    def test_reaction_iterator(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        entries_box = EntriesBox(molecule_entries)
        reaction_iterator = ReactionIterator(entries_box, single_elem_interm_ignore=[])
        reactions = []

        for reaction in reaction_iterator:
            reactions.append((reaction[0], reaction[1]))

        result = frozenset(reactions)

        # ronalds concerteds is a json dump of reactions since we can't serialize frozensets to json
        ronalds_concerteds_lists = loadfn(
            os.path.join(test_dir, "ronalds_concerteds.json")
        )
        ronalds_concerteds = frozenset(
            [tuple([tuple(x[0]), tuple(x[1])]) for x in ronalds_concerteds_lists]
        )

        assert result == ronalds_concerteds

    def test_filter(self):
        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        entries_box = EntriesBox(molecule_entries)

        iter_unfiltered = ReactionIterator(entries_box, single_elem_interm_ignore=[])
        rxns_unfiltered = sorted([e for e in iter_unfiltered])
        unfiltered_reference = loadfn(
            os.path.join(test_dir, "unfiltered_rxns_sorted.json")
        )
        unfiltered_reference = [
            (tuple(x[0]), tuple(x[1]), x[2]) for x in unfiltered_reference
        ]
        assert rxns_unfiltered == unfiltered_reference

        iter_filtered = ReactionIterator(
            entries_box,
            single_elem_interm_ignore=[],
            filter_concerted_metal_coordination=True,
        )
        rxns_filtered = sorted([e for e in iter_filtered])
        filtered_reference = loadfn(os.path.join(test_dir, "filtered_rxns_sorted.json"))
        filtered_reference = [
            (tuple(x[0]), tuple(x[1]), x[2]) for x in filtered_reference
        ]
        assert rxns_filtered == filtered_reference


class TestReactionGenerator(PymatgenTest):
    def test_build(self):
        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        entries_box = EntriesBox(molecule_entries)
        RG = ReactionGenerator(entries_box)
        RG.build()
        assert RG.redox_c == 15
        assert RG.inter_c == 32
        assert RG.intra_c == 2
        assert RG.coord_c == 10
        assert len(RG.graph.edges) == 320
        assert len(RG.graph.nodes) == 164

    def test_parse_reaction_node(self):

        nodes = ["19+32,673", "41,992", "1+652,53+40", "4,6+5"]
        node_prod_react = []

        for node in nodes:
            r_p = ReactionGenerator.parse_reaction_node(node)
            node_prod_react.append(r_p)

        assert node_prod_react == [
            ([19, 32], [673]),
            ([41], [992]),
            ([1, 652], [40, 53]),
            ([4], [5, 6]),
        ]

    def test_generate_node_string(self):

        react_prod = [
            ([19, 32], [673]),
            ([41], [992]),
            ([1, 652], [40, 53]),
            ([4], [5, 6]),
        ]
        node_strings = []

        for rxn in react_prod:
            node_str = ReactionGenerator.generate_node_string(rxn[0], rxn[1])
            node_strings.append(node_str)

        assert node_strings == ["19+32,673", "41,992", "1+652,40+53", "4,5+6"]

    def test_concerted_reaction_filter(self):
        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        entries_box = EntriesBox(molecule_entries)
        RG = ReactionGenerator(entries_box)
        r, r_node = RG.concerted_reaction_filter("6,2+7", "2+1,3")
        assert r == [[1, 6], [3, 7]]
        assert r_node == [[1, 6], [3, 7], ["6,2+7", "2+1,3"]]
        r, r_node = RG.concerted_reaction_filter("2+1,3+10", "6,2+7")
        assert r == None
        assert r_node == None

    # def test_build_matrix(self):

    #     with open(
    #         os.path.join(
    #             test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
    #         ),
    #         "rb",
    #     ) as input:
    #         RN_loaded = pickle.load(input)

    #     with open(os.path.join(test_dir, "RN_matrix_build.pkl"), "rb") as handle:
    #         loaded_matrix = pickle.load(handle)

    #     with open(
    #         os.path.join(test_dir, "RN_matrix_inverse_build.pkl"), "rb"
    #     ) as handle:
    #         loaded_inverse_matrix = pickle.load(handle)

    #     RN_loaded.build_matrix()

    #     def matrix_assert(computed, loaded):
    #         for r in loaded:  # iterate over reactants (rows)
    #             for p in loaded[r]:  # iterate over products (cols)
    #                 try:
    #                     for k in range(
    #                         len(loaded[r][p])
    #                     ):  # iterate over reactions (element, list form)
    #                         pr_free = loaded[r][p][k][0].replace("PR_", "")  # node name
    #                         assert computed[r][p][k][1] == loaded[r][p][k][1]  # weight
    #                         assert (
    #                             computed[r][p][k][2] == loaded[r][p][k][2]
    #                         )  # rxn type (elem/concert)
    #                         try:
    #                             assert computed[r][p][k][0] == pr_free
    #                         except AssertionError:
    #                             match = False
    #                             for perm in permutations(
    #                                 pr_free.split(",")[0].split("+")
    #                             ):
    #                                 permuted_rxn = (
    #                                     "+".join(perm) + "," + pr_free.split(",")[1]
    #                                 )
    #                                 match = permuted_rxn == computed[r][p][k][0]
    #                                 if match:
    #                                     break
    #                             assert match
    #                 except KeyError:
    #                     print(loaded[i])

    #         return loaded

    #     matrix_assert(RN_loaded.matrix, loaded_matrix)
    #     matrix_assert(RN_loaded.matrix_inverse, loaded_inverse_matrix)

    # def test_identify_concerted_rxns_via_intermediates(self):
    #     # load RN
    #     with open(
    #         os.path.join(
    #             test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
    #         ),
    #         "rb",
    #     ) as input:
    #         RN_loaded = pickle.load(input)

    #     RN_loaded.matrix = None
    #     (
    #         v3_unique_iter1,
    #         v3_all_iter1,
    #     ) = RN_loaded.identify_concerted_rxns_via_intermediates(
    #         RN_loaded, single_elem_interm_ignore=[], update_matrix=True
    #     )
    #     unique_reactions = loadfn(
    #         os.path.join(
    #             test_dir,
    #             "identify_concerted_rxns_via_intermediates_v3_all_unique_reactions.json",
    #         )
    #     )
    #     self.assertEqual(len(v3_unique_iter1), len(unique_reactions))
    #     v3_unique_computed_set = set(map(lambda x: repr(x), v3_unique_iter1))
    #     v3_unique_loaded_set = set(map(lambda x: repr(x), unique_reactions))
    #     self.assertEqual(v3_unique_computed_set, v3_unique_loaded_set)
    #     reactions_with_intermediates = loadfn(
    #         os.path.join(
    #             test_dir,
    #             "identify_concerted_rxns_via_intermediates_v3_with_intermediate_nodes.json",
    #         )
    #     )
    #     self.assertEqual(len(v3_all_iter1), len(reactions_with_intermediates))
    #     for i in range(len(reactions_with_intermediates)):
    #         for j in range(len(reactions_with_intermediates[i])):
    #             for k in range(len(reactions_with_intermediates[i][j])):
    #                 for l in range(len(reactions_with_intermediates[i][j][k])):
    #                     if isinstance(reactions_with_intermediates[i][j][k][l], str):
    #                         el = reactions_with_intermediates[i][j][k][l]
    #                         el = el.replace("PR_", "")
    #                         sorted_rcts = sorted(el.split(",")[0].split("+"))
    #                         sorted_pros = sorted(el.split(",")[1].split("+"))
    #                         reactions_with_intermediates[i][j][k][l] = (
    #                             "+".join(sorted_rcts) + "," + "+".join(sorted_pros)
    #                         )
    #     for i in range(len(v3_all_iter1)):
    #         for j in range(len(v3_all_iter1[i])):
    #             for k in range(len(v3_all_iter1[i][j])):
    #                 for l in range(len(v3_all_iter1[i][j][k])):
    #                     if isinstance(v3_all_iter1[i][j][k][l], str):
    #                         el = v3_all_iter1[i][j][k][l]
    #                         sorted_rcts = sorted(el.split(",")[0].split("+"))
    #                         sorted_pros = sorted(el.split(",")[1].split("+"))
    #                         v3_all_iter1[i][j][k][l] = (
    #                             "+".join(sorted_rcts) + "," + "+".join(sorted_pros)
    #                         )
    #     v3_all_computed_set = set(map(lambda x: repr(x), v3_all_iter1))
    #     v3_all_loaded_set = set(map(lambda x: repr(x), reactions_with_intermediates))
    #     try:
    #         self.assertEqual(v3_all_computed_set, v3_all_loaded_set)
    #     except AssertionError:
    #         c_minus_l = v3_all_computed_set - v3_all_loaded_set
    #         l_minus_c = v3_all_loaded_set - v3_all_computed_set
    #         self.assertEqual(len(c_minus_l), len(l_minus_c))
    #         print(len(c_minus_l))

    # def test_identify_concerted_rxns_for_specific_intermediate(self):

    #     with open(
    #         os.path.join(
    #             test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"
    #         ),
    #         "rb",
    #     ) as input:
    #         RN_loaded = pickle.load(input)
    #     RN_loaded.matrix = None
    #     (
    #         reactions,
    #         reactions_with_nodes,
    #     ) = RN_loaded.identify_concerted_rxns_for_specific_intermediate(
    #         RN_loaded.entries_list[1],
    #         RN_loaded,
    #         single_elem_interm_ignore=[],
    #         update_matrix=False,
    #     )
    #     unique_reactions = set(map(lambda x: repr(x), reactions))
    #     self.assertEqual(len(unique_reactions), 6901)


if __name__ == "__main__":
    unittest.main()
