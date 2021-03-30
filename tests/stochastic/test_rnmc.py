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
from mrnet.stochastic.serialize import (
    SerializedReactionNetwork,
    serialize_simulation_parameters,
    find_mol_entry_from_xyz_and_charge,
    run_simulator,
)
from mrnet.stochastic.analyze import SimulationAnalyzer, load_analysis

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
    def test_rnmc(self):

        molecule_entries = loadfn(os.path.join(test_dir, "ronalds_MoleculeEntry.json"))

        li_plus_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "Li.xyz")), 1
        )

        ec_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "EC.xyz")), 0
        )

        ledc_mol_entry = find_mol_entry_from_xyz_and_charge(
            molecule_entries, (os.path.join(test_dir, "EC.xyz")), 0
        )

        network_folder_1 = "/tmp/RNMC_network_1"
        network_folder_2 = "/tmp/RNMC_network_2"
        param_folder = "/tmp/RNMC_params"

        initial_state_data_1 = [(li_plus_mol_entry, 300), (ec_mol_entry, 30)]
        initial_state_data_2 = [(li_plus_mol_entry, 30), (ec_mol_entry, 300)]

        reaction_generator = ReactionGenerator(molecule_entries)
        rnsd = SerializedReactionNetwork(reaction_generator)
        rnsd.serialize(network_folder_1, initial_state_data_1)
        rnsd.serialize(network_folder_2, initial_state_data_2)
        serialize_simulation_parameters(param_folder, number_of_threads=4)

        run_simulator(network_folder_1, param_folder)
        run_simulator(network_folder_2, param_folder)

        sa_1 = load_analysis(network_folder_1)
        sa_1.generate_pathway_report(ledc_mol_entry, 10)
        sa_1.generate_consumption_report(ledc_mol_entry)
        sa_1.generate_reaction_tally_report()

        sa_2 = load_analysis(network_folder_2)
        sa_2.generate_pathway_report(ledc_mol_entry, 10)
        sa_2.generate_consumption_report(ledc_mol_entry)
        sa_2.generate_reaction_tally_report()

        os.system("rm -r " + network_folder_1)
        os.system("rm -r " + network_folder_2)
        os.system("rm -r " + param_folder)
