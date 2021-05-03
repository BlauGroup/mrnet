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

from mrnet.network.reaction_generation import ReactionIterator
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
        reaction_generator = ReactionIterator(
            molecule_entries, single_elem_interm_ignore=[]
        )
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

        # make test idempotent after failure
        network_folder_1 = "/tmp/RNMC_network_1"
        network_folder_2 = "/tmp/RNMC_network_2"
        param_folder = "/tmp/RNMC_params"

        os.system("rm -r " + network_folder_1)
        os.system("rm -r " + network_folder_2)
        os.system("rm -r " + param_folder)

        initial_state_data_1 = [(li_plus_mol_entry, 300), (ec_mol_entry, 30)]
        initial_state_data_2 = [(li_plus_mol_entry, 30), (ec_mol_entry, 300)]

        reaction_generator = ReactionIterator(molecule_entries)

        # for large networks, you want to use shard_size=2000000
        SerializeNetwork(network_folder_1, reaction_generator, shard_size=100)
        network_updater = NetworkUpdater(network_folder_1)
        # recompute all rates using a fixed constant barrier
        network_updater.recompute_all_rates(ROOM_TEMP, 0.3)

        # check that no duplicates got inserted
        assert len(network_updater.find_duplicates()) == 0

        # serializing is expensive, so we only want to do it once
        # instead, for reaction_network_2 we symlink the database into the folder
        clone_database(network_folder_1, network_folder_2)

        serialize_initial_state(
            network_folder_1, molecule_entries, initial_state_data_1
        )
        serialize_initial_state(
            network_folder_2, molecule_entries, initial_state_data_2
        )
        serialize_simulation_parameters(param_folder, number_of_threads=4)

        run_simulator(network_folder_1, param_folder)
        run_simulator(network_folder_2, param_folder)

        sa_1 = SimulationAnalyzer(network_folder_1, molecule_entries)
        sa_1.generate_pathway_report(ledc_mol_entry, 10)
        sa_1.generate_consumption_report(ledc_mol_entry)
        sa_1.generate_reaction_tally_report(10)
        profiles_1 = sa_1.generate_time_dep_profiles()
        states_1 = sa_1.final_state_analysis(profiles_1["final_states"])
        rxn_counts_1 = sa_1.rank_reaction_counts()

        sa_2 = SimulationAnalyzer(network_folder_2, molecule_entries)
        sa_2.generate_pathway_report(ledc_mol_entry, 10)
        sa_2.generate_consumption_report(ledc_mol_entry)
        sa_2.generate_reaction_tally_report(10)
        profiles_2 = sa_2.generate_time_dep_profiles()
        states_2 = sa_2.final_state_analysis(profiles_2["final_states"])
        rxn_counts_2 = sa_2.rank_reaction_counts()

        # update rates from a list

        # set specific rates
        network_updater.update_rates([(113, 2.0), (40, 3.0)])

        os.system("rm -r " + network_folder_1)
        os.system("rm -r " + network_folder_2)
        os.system("rm -r " + param_folder)
