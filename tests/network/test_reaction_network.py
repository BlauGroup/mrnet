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
from mrnet.network.reaction_network import (
    ReactionPath,
    ReactionNetwork,
    path_finding_wrapper,
)
from mrnet.network.reaction_generation import ReactionIterator
from mrnet.stochastic.serialize import find_mol_entry_from_xyz_and_charge

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


class DanielTest(PymatgenTest):
    def test_path_finding(self):
        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        li_plus_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "Li.xyz")), 1
        )

        ec_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "EC.xyz")), 0
        )

        ledc_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "LEDC.xyz")), 0
        )

        result = path_finding_wrapper(
            molecule_entries, [li_plus_mol_entry, ec_mol_entry], ledc_mol_entry
        )

        dumpfn(result, "/tmp/lol")
        result_canonicalized = loadfn("/tmp/lol")

        expected = loadfn(os.path.join(test_dir, "ronalds_PRs.json"))

        assert result_canonicalized == expected


if __name__ == "__main__":
    unittest.main()
