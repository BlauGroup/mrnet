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

test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


class TestReactionGenerator(PymatgenTest):
    def test_reaction_generator(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        entries_box = EntriesBox(molecule_entries)
        reaction_generator = ReactionIterator(entries_box, single_elem_interm_ignore=[])
        reactions = []

        for reaction in reaction_generator:
            reactions.append(
                (
                    tuple([int(r) for r in reaction.reactant_indices]),
                    tuple([int(r) for r in reaction.product_indices]),
                )
            )

        result = frozenset(reactions)

        # ronalds concerteds is a json dump of reactions since we can't serialize frozensets to json
        ronalds_concerteds_lists = loadfn(
            os.path.join(test_dir, "ronalds_concerteds.json")
        )
        ronalds_concerteds = frozenset(
            [tuple([tuple(x[0]), tuple(x[1])]) for x in ronalds_concerteds_lists]
        )

        assert result == ronalds_concerteds
