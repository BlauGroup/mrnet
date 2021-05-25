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
from mrnet.network.reaction_generation import ReactionIterator, EntriesBox
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

try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None

__author__ = "Daniel Barter"

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


class TestReactionGenerator(PymatgenTest):
    def test_reaction_generator(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        entries_box = EntriesBox(molecule_entries)
        reaction_generator = ReactionIterator(entries_box, single_elem_interm_ignore=[])
        reactions = []

        for reaction in reaction_generator:
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
