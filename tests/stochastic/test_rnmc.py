import subprocess
import unittest
import copy
import pickle
import math
import os

import numpy as np
from scipy.constants import N_A
from monty.serialization import loadfn
from pymatgen.util.testing import PymatgenTest

from mrnet.network.reaction_generation import ReactionGenerator
from mrnet.stochastic.rnmc import (
    SerializedReactionNetwork,
    find_mol_entry_from_xyz_and_charge,
    run,
)

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


class RNMC(PymatgenTest):
    def test_reaction_network_serialization(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))
        reaction_generator = ReactionGenerator(molecule_entries)

        li_plus_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "Li.xyz")), 1
        )

        ec_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "EC.xyz")), 0
        )

        network_folder = "./RNMC_network"
        param_folder = "./RNMC_params"

        initial_state_data = [(li_plus_mol_entry, 30), (ec_mol_entry, 30)]

        rnsd = ReactionNetworkSerializationData(
            reaction_generator,
            initial_state_data,
            network_folder,
            param_folder,
            logging=False,
        )

        self.assertEqual(rnsd.number_of_reactions, 212)

    def test_reaction_network_serialization(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))

        li_plus_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "Li.xyz")), 1
        )

        ec_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "EC.xyz")), 0
        )

        network_folder = "./RNMC_network"
        param_folder = "./RNMC_params"

        initial_state_data = [(li_plus_mol_entry, 30), (ec_mol_entry, 30)]

        run(
            molecule_entries,
            initial_state_data,
            network_folder,
            param_folder
        )

        os.system("rm -r " + network_folder)
        os.system("rm -r " + param_folder)
