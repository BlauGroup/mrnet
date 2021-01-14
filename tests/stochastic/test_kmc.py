import os
import unittest
import copy
import pickle
import math

import numpy as np
from scipy.constants import N_A

from pymatgen.util.testing import PymatgenTest

from mrnet.stochastic.kmc import (
    initialize_simulation,
    kmc_simulate,
    update_state,
    get_coordination,
    KmcDataAnalyzer,
)

try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


class TestKMCReactionPropagatorFxns(PymatgenTest):
    def setUp(self):
        """Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O
        """
        self.volume = 10 ** -24  # m^3

        # 100 molecules each of H2O, H2, O2, OH-, H+
        self.num_mols = int(100)
        self.concentration = self.num_mols / N_A / self.volume / 1000

        if ob:
            pickle_in = open(os.path.join(test_dir, "h2o_test_network.pickle"), "rb")
            self.reaction_network = pickle.load(pickle_in)
            pickle_in.close()

            # Only H2O, H2, O2 present initially
            self.initial_conditions = {
                "h2o": self.concentration,
                "h2": self.concentration,
                "o2": self.concentration,
                "oh-": self.concentration,
                "h+": self.concentration,
            }
            self.initial_cond_mols = dict()
            self.num_species = len(self.reaction_network.entries_list)
            self.num_reactions = len(self.reaction_network.reactions)
            self.reactants = np.ones((self.num_reactions, 2), dtype=int) * -1
            self.products = np.ones((self.num_reactions, 2), dtype=int) * -1
            self.rate_constants = np.zeros(2 * self.num_reactions)
            self.coord_array = np.zeros(2 * self.num_reactions)
            self.molid_ind_mapping = dict()
            self.initial_state = list()
            for ind, mol in enumerate(self.reaction_network.entries_list):
                if mol.entry_id in self.initial_conditions:
                    self.initial_state.append(self.num_mols)
                else:
                    self.initial_state.append(0)
                self.molid_ind_mapping[mol.entry_id] = ind
            self.initial_state = np.array(self.initial_state)

            species_rxn_mapping_list = [[] for i in range(self.num_species)]

            for ind, spec in enumerate(self.reaction_network.entries_list):
                self.molid_ind_mapping[spec.entry_id] = ind
            # print('molid_ind mapping:', self.molid_ind_mapping)
            # construct the product and reactants arrays

            for mol_id, conc in self.initial_conditions.items():
                self.initial_cond_mols[self.molid_ind_mapping[mol_id]] = self.num_mols

            for ind, reaction in enumerate(self.reaction_network.reactions):
                num_reactants_for = list()
                num_reactants_rev = list()
                for idx, react in enumerate(reaction.reactants):
                    mol_ind = self.molid_ind_mapping[react.entry_id]
                    self.reactants[ind, idx] = mol_ind
                    num_reactants_for.append(self.initial_state[mol_ind])
                    species_rxn_mapping_list[mol_ind].append(2 * ind)
                for idx, prod in enumerate(reaction.products):
                    mol_ind = self.molid_ind_mapping[prod.entry_id]
                    self.products[ind, idx] = mol_ind
                    num_reactants_rev.append(self.initial_state[mol_ind])
                    species_rxn_mapping_list[mol_ind].append(2 * ind + 1)

                reaction.set_rate_constant()
                self.rate_constants[2 * ind] = reaction.k_A
                self.rate_constants[2 * ind + 1] = reaction.k_B
                # set up coordination array
                if len(reaction.reactants) == 1:
                    self.coord_array[2 * ind] = num_reactants_for[0]
                elif (len(reaction.reactants) == 2) and (
                    reaction.reactants[0] == reaction.reactants[1]
                ):
                    self.coord_array[2 * ind] = num_reactants_for[0] * (
                        num_reactants_for[0] - 1
                    )
                elif (len(reaction.reactants) == 2) and (
                    reaction.reactants[0] != reaction.reactants[1]
                ):
                    self.coord_array[2 * ind] = (
                        num_reactants_for[0] * num_reactants_for[1]
                    )
                else:
                    raise RuntimeError(
                        "Only single and bimolecular reactions supported by this simulation"
                    )
                # For reverse reaction
                if len(reaction.products) == 1:
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0]
                elif (len(reaction.products) == 2) and (
                    reaction.products[0] == reaction.products[1]
                ):
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0] * (
                        num_reactants_rev[0] - 1
                    )
                elif (len(reaction.products) == 2) and (
                    reaction.products[0] != reaction.products[1]
                ):
                    self.coord_array[2 * ind + 1] = (
                        num_reactants_rev[0] * num_reactants_rev[1]
                    )
                else:
                    raise RuntimeError(
                        "Only single and bimolecular reactions supported by this simulation"
                    )

            self.propensities = np.multiply(self.coord_array, self.rate_constants)
            # Set up molind_rxn_mapping
            spec_rxn_map_lengths = [
                len(rxn_list) for rxn_list in species_rxn_mapping_list
            ]
            max_map_length = max(spec_rxn_map_lengths)
            self.species_rxn_mapping = (
                np.ones((self.num_species, max_map_length), dtype=int) * -1
            )
            for ind, rxn_list in enumerate(species_rxn_mapping_list):
                if len(rxn_list) == max_map_length:
                    self.species_rxn_mapping[ind, :] = rxn_list
                else:
                    self.species_rxn_mapping[ind, : len(rxn_list)] = rxn_list

            for rxn_ind, reaction in enumerate(self.reaction_network.reactions):
                react_id_list = [r.entry_id for r in reaction.reactants]
                prod_id_list = [p.entry_id for p in reaction.products]

    def tearDown(self) -> None:
        if ob:
            del self.volume
            del self.num_mols
            del self.concentration
            del self.reaction_network
            del self.initial_conditions
            del self.initial_cond_mols
            del self.products
            del self.reactants
            del self.rate_constants
            del self.initial_state
            del self.coord_array
            del self.num_species
            del self.num_reactions
            del self.species_rxn_mapping
            del self.propensities
            del self.molid_ind_mapping

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_initialize_simulation(self):
        # initialize the simulation
        [
            initial_state,
            initial_state_dict,
            species_rxn_mapping,
            reactant_array,
            product_array,
            coord_array,
            rate_constants,
            propensities,
            molid_index_mapping,
        ] = initialize_simulation(
            self.reaction_network, self.initial_conditions, self.volume
        )
        # Verify initial state
        num_species = len(self.reaction_network.entries_list)
        exp_initial_state = np.array([0 for i in range(num_species)])
        exp_initial_state[2] = self.num_mols  # h+
        exp_initial_state[3] = self.num_mols  # oh-
        exp_initial_state[7] = self.num_mols  # h2
        exp_initial_state[10] = self.num_mols  # h2o
        exp_initial_state[16] = self.num_mols  # o2
        self.assertArrayAlmostEqual(exp_initial_state, initial_state)
        self.assertArrayAlmostEqual(self.initial_state, initial_state)
        self.assertArrayAlmostEqual(self.products, product_array)
        self.assertArrayAlmostEqual(self.reactants, reactant_array)
        self.assertArrayAlmostEqual(self.rate_constants, rate_constants)
        self.assertArrayAlmostEqual(self.species_rxn_mapping, species_rxn_mapping)
        self.assertArrayAlmostEqual(self.propensities, propensities)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_update_state(self):
        # Reactions of interest:
        # 1) h2 <--> h2+
        # 2) h2o- <--> h2o
        # 3) h2 <--> h- + h+
        reactions_sequence = list()
        for ind, reaction in enumerate(self.reaction_network.reactions):
            reactants = [react.entry_id for react in reaction.reactants]
            products = [prod.entry_id for prod in reaction.products]
            if (
                ((["h2"] == reactants) and (["h2+"] == products))
                or ((["h2o"] == reactants) and (["h2o-"] == products))
                or ((["h2"] == reactants) and (["h-", "h+"] == products))
            ):
                reactions_sequence.extend([2 * ind, 2 * ind + 1, 2 * ind])
            elif (
                ((["h2+"] == reactants) and (["h2"] == products))
                or ((["h2o-"] == reactants) and (["h2o"] == products))
                or ((["h-", "h+"] == reactants) and (["h"] == products))
            ):
                reactions_sequence.extend([2 * ind + 1, 2 * ind, 2 * ind + 1])

        num_iterations = 10
        state = np.array(self.initial_state)
        for i in range(num_iterations):
            for rxn_ind in reactions_sequence:
                if rxn_ind % 2:
                    reverse = True
                else:
                    reverse = False
                converted_rxn_ind = math.floor(rxn_ind / 2)
                state = update_state(
                    self.reactants, self.products, state, converted_rxn_ind, reverse
                )

        expected_state = [0 for i in range(self.num_species)]
        expected_state[self.molid_ind_mapping["h-"]] = num_iterations
        expected_state[self.molid_ind_mapping["h+"]] = self.num_mols + num_iterations
        expected_state[self.molid_ind_mapping["oh-"]] = self.num_mols
        expected_state[self.molid_ind_mapping["h2"]] = (
            self.num_mols - 2 * num_iterations
        )
        expected_state[self.molid_ind_mapping["h2+"]] = num_iterations
        expected_state[self.molid_ind_mapping["h2o-"]] = num_iterations
        expected_state[self.molid_ind_mapping["h2o"]] = self.num_mols - num_iterations
        expected_state[self.molid_ind_mapping["o2"]] = self.num_mols
        self.assertEqual(list(state), expected_state)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_get_coordination(self):
        # Reactions of interest:
        # h2 <--> h2+
        # h2o <--> oh- + h+
        state = np.array(self.initial_state)
        for ind, reaction in enumerate(self.reaction_network.reactions):
            run_test = 0
            reactants = [react.entry_id for react in reaction.reactants]
            products = [prod.entry_id for prod in reaction.products]
            if (["h2"] == reactants) and (["h2+"] == products):
                run_test = 1
                reaction_sequence = [2 * ind, 2 * ind + 1]
            elif (["h2+"] == reactants) and (["h2"] == products):
                run_test = 1
                reaction_sequence = [2 * ind + 1, 2 * ind]
            elif (["h2o"] == reactants) and (["oh-", "h+"] == products):
                run_test = 2
                reaction_sequence = [2 * ind, 2 * ind + 1]
            elif (["oh-", "h+"] == reactants) and (["h2o"] == products):
                run_test = 2
                reaction_sequence = [2 * ind + 1, 2 * ind]

            if run_test == 0:
                continue
            else:
                coords = list()
                if run_test == 1:  # testing h2 <--> h2+
                    expected_coords = [self.num_mols, 0]
                elif run_test == 2:  # testing h2o <--> oh- + h+
                    expected_coords = [self.num_mols, self.num_mols * self.num_mols]

                for rxn_ind in reaction_sequence:
                    if rxn_ind % 2:
                        reverse = True
                    else:
                        reverse = False
                    converted_rxn_ind = math.floor(rxn_ind / 2)
                    coords.append(
                        get_coordination(
                            self.reactants,
                            self.products,
                            state,
                            converted_rxn_ind,
                            reverse,
                        )
                    )

                self.assertEqual(expected_coords, coords)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_kmc_simulate(self):
        t_steps = 1
        iterations = 10000
        reaction_history = list()
        time_steps = list()
        reaction_frequency = [0 for i in range(2 * self.num_reactions)]
        total_propensity = np.sum(self.propensities)
        exp_tau = 1 / total_propensity  # expectation value of first time step
        rxn_probability = (
            self.propensities / total_propensity
        )  # expected frequencies of each reaction
        # for rxn in self.reaction_network.reactions:
        for i in range(iterations):
            # run simulation with initial conditions, 1 time step
            sim_data = kmc_simulate(
                t_steps,
                self.coord_array,
                self.rate_constants,
                self.propensities,
                self.species_rxn_mapping,
                self.reactants,
                self.products,
                np.array(self.initial_state),
            )
            if (t_steps != len(sim_data[0])) or (t_steps != len(sim_data[1])):
                raise RuntimeError(
                    "There are more than the specified time steps for this simulation."
                )
            reaction_history.append(int(sim_data[0][0]))
            time_steps.append(sim_data[1][0])

        reaction_history = np.array(reaction_history)
        time_steps = np.array(time_steps)
        avg_tau = np.average(time_steps)
        self.assertAlmostEqual(avg_tau, exp_tau, places=7)


class TestKmcDataAnalyzer(PymatgenTest):
    def setUp(self):
        """Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O
        """
        self.volume = 10 ** -24  # m^3

        # 100 molecules each of H2O, H2, O2, OH-, H+
        self.num_mols = int(100)
        self.concentration = self.num_mols / N_A / self.volume / 1000

        if ob:
            pickle_in = open(os.path.join(test_dir, "h2o_test_network.pickle"), "rb")
            self.reaction_network = pickle.load(pickle_in)
            pickle_in.close()

            # Define initially present molecules
            self.initial_conditions = {
                "h2o": self.concentration,
                "h2": self.concentration,
                "o2": self.concentration,
                "oh-": self.concentration,
                "h+": self.concentration,
            }
            self.initial_cond_mols = dict()
            self.num_species = len(self.reaction_network.entries_list)
            self.num_reactions = len(self.reaction_network.reactions)
            self.reactants = np.ones((self.num_reactions, 2), dtype=int) * -1
            self.products = np.ones((self.num_reactions, 2), dtype=int) * -1
            self.rate_constants = np.zeros(2 * self.num_reactions)
            self.coord_array = np.zeros(2 * self.num_reactions)
            self.molid_ind_mapping = dict()
            self.initial_state = list()
            for ind, mol in enumerate(self.reaction_network.entries_list):
                if mol.entry_id in self.initial_conditions:
                    self.initial_state.append(self.num_mols)
                else:
                    self.initial_state.append(0)
                self.molid_ind_mapping[mol.entry_id] = ind
            self.initial_state = np.array(self.initial_state)

            species_rxn_mapping_list = [[] for i in range(self.num_species)]
            # print('molid_ind mapping:', self.molid_ind_mapping)
            # construct the product and reactants arrays

            for mol_id, conc in self.initial_conditions.items():
                self.initial_cond_mols[self.molid_ind_mapping[mol_id]] = self.num_mols

            for ind, reaction in enumerate(self.reaction_network.reactions):
                num_reactants_for = list()
                num_reactants_rev = list()
                for idx, react in enumerate(reaction.reactants):
                    mol_ind = self.molid_ind_mapping[react.entry_id]
                    self.reactants[ind, idx] = mol_ind
                    num_reactants_for.append(self.initial_state[mol_ind])
                    species_rxn_mapping_list[mol_ind].append(2 * ind)
                for idx, prod in enumerate(reaction.products):
                    mol_ind = self.molid_ind_mapping[prod.entry_id]
                    self.products[ind, idx] = mol_ind
                    num_reactants_rev.append(self.initial_state[mol_ind])
                    species_rxn_mapping_list[mol_ind].append(2 * ind + 1)
                reaction.set_rate_constant()
                self.rate_constants[2 * ind] = reaction.k_A
                self.rate_constants[2 * ind + 1] = reaction.k_B
                # set up coordination array
                if len(reaction.reactants) == 1:
                    self.coord_array[2 * ind] = num_reactants_for[0]
                elif (len(reaction.reactants) == 2) and (
                    reaction.reactants[0] == reaction.reactants[1]
                ):
                    self.coord_array[2 * ind] = num_reactants_for[0] * (
                        num_reactants_for[0] - 1
                    )
                elif (len(reaction.reactants) == 2) and (
                    reaction.reactants[0] != reaction.reactants[1]
                ):
                    self.coord_array[2 * ind] = (
                        num_reactants_for[0] * num_reactants_for[1]
                    )
                else:
                    raise RuntimeError(
                        "Only single and bimolecular reactions supported by this simulation"
                    )
                # For reverse reaction
                if len(reaction.products) == 1:
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0]
                elif (len(reaction.products) == 2) and (
                    reaction.products[0] == reaction.products[1]
                ):
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0] * (
                        num_reactants_rev[0] - 1
                    )
                elif (len(reaction.products) == 2) and (
                    reaction.products[0] != reaction.products[1]
                ):
                    self.coord_array[2 * ind + 1] = (
                        num_reactants_rev[0] * num_reactants_rev[1]
                    )
                else:
                    raise RuntimeError(
                        "Only single and bimolecular reactions supported by this simulation"
                    )

            self.propensities = np.multiply(self.coord_array, self.rate_constants)
            # Set up molind_rxn_mapping
            spec_rxn_map_lengths = [
                len(rxn_list) for rxn_list in species_rxn_mapping_list
            ]
            max_map_length = max(spec_rxn_map_lengths)
            self.species_rxn_mapping = (
                np.ones((self.num_species, max_map_length), dtype=int) * -1
            )
            for ind, rxn_list in enumerate(species_rxn_mapping_list):
                if len(rxn_list) == max_map_length:
                    self.species_rxn_mapping[ind, :] = rxn_list
                else:
                    self.species_rxn_mapping[ind, : len(rxn_list)] = rxn_list

            for rxn_ind, reaction in enumerate(self.reaction_network.reactions):
                react_id_list = [r.entry_id for r in reaction.reactants]
                prod_id_list = [p.entry_id for p in reaction.products]
                if (["h2o"] == react_id_list) and (["h2o-"] == prod_id_list):
                    self.rxn_a = 2 * rxn_ind
                    self.rxn_b = 2 * rxn_ind + 1
                    break
                elif (["h2o-"] == react_id_list) and (["h2o"] == prod_id_list):
                    self.rxn_b = 2 * rxn_ind
                    self.rxn_a = 2 * rxn_ind + 1
                    break
            # setting up a sequence of reaction histories
            rxns_1 = np.append(
                self.rxn_a * np.ones(6, dtype=int), self.rxn_b * np.ones(6, dtype=int)
            )
            rxns_2 = np.append(
                self.rxn_a * np.ones(8, dtype=int), self.rxn_b * np.ones(4, dtype=int)
            )
            rxns_3 = np.append(
                self.rxn_a * np.ones(3, dtype=int), self.rxn_b * np.ones(3, dtype=int)
            )
            rxns_3 = np.append(rxns_3, rxns_3)
            self.reaction_history = [rxns_1, rxns_2, rxns_3]
            # time increment between each reaction is 1
            self.time_history = [np.ones(12) for i in range(3)]
            self.analyzer = KmcDataAnalyzer(
                self.reaction_network,
                self.molid_ind_mapping,
                self.species_rxn_mapping,
                self.initial_cond_mols,
                self.products,
                self.reactants,
                self.reaction_history,
                self.time_history,
            )

    def tearDown(self) -> None:
        if ob:
            del self.volume
            del self.num_mols
            del self.concentration
            del self.reaction_network
            del self.initial_conditions
            del self.initial_cond_mols
            del self.products
            del self.reactants
            del self.rate_constants
            del self.initial_state
            del self.coord_array
            del self.num_species
            del self.num_reactions
            del self.species_rxn_mapping
            del self.propensities
            del self.molid_ind_mapping
            del self.reaction_history
            del self.time_history
            del self.analyzer
            del self.rxn_b
            del self.rxn_a

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate_time_dep_profiles_intermediates_analysis_final_state_analysis(
        self,
    ):
        exp_trajectories = list()
        initial_cond = dict()
        for mol_id in self.initial_conditions:
            initial_cond[self.molid_ind_mapping[mol_id]] = [(0.0, self.num_mols)]
            if (mol_id != "h2o") and (mol_id != "h2o-"):
                initial_cond[self.molid_ind_mapping[mol_id]].append(
                    (12.0, self.num_mols)
                )
        # print(initial_cond)
        for i in range(3):
            exp_trajectories.append(copy.deepcopy(initial_cond))

        # first sequence of reactions: [a, a, a, a, a, a, b, b, b, b, b, b]
        # generate expected trajectories
        for i in range(1, 13):
            if i <= 6:
                exp_trajectories[0][10].append((float(i), self.num_mols - i))
            else:
                exp_trajectories[0][10].append((float(i), self.num_mols - 12 + i))
        exp_trajectories[0][10].append(exp_trajectories[0][10][-1])

        exp_trajectories[0][9] = list()
        for i in range(13):
            if i <= 6:
                exp_trajectories[0][9].append((float(i), i))
            else:
                exp_trajectories[0][9].append((float(i), 12 - i))
        exp_trajectories[0][9].append(exp_trajectories[0][9][-1])

        # second sequence: [a, a, a, a, a, a, a, a, b, b, b, b]
        for i in range(1, 13):
            if i <= 8:
                exp_trajectories[1][10].append((float(i), self.num_mols - i))
            else:
                exp_trajectories[1][10].append((float(i), self.num_mols - 16 + i))
        exp_trajectories[1][10].append(exp_trajectories[1][10][-1])

        exp_trajectories[1][9] = list()
        for i in range(13):
            if i <= 8:
                exp_trajectories[1][9].append((float(i), i))
            else:
                exp_trajectories[1][9].append((float(i), 16 - i))
        exp_trajectories[1][9].append(exp_trajectories[1][9][-1])

        # third sequence: [a, a, a, b, b, b, a, a, a, b, b, b]
        for i in range(1, 13):
            if i <= 3:
                exp_trajectories[2][10].append((float(i), self.num_mols - i))
            elif i <= 6:
                exp_trajectories[2][10].append((float(i), self.num_mols - 6 + i))
            elif i <= 9:
                exp_trajectories[2][10].append((float(i), self.num_mols + 6 - i))
            else:
                exp_trajectories[2][10].append((float(i), self.num_mols - 12 + i))
        exp_trajectories[2][10].append(exp_trajectories[2][10][-1])

        exp_trajectories[2][9] = list()
        for i in range(13):
            if i <= 3:
                exp_trajectories[2][9].append((float(i), i))
            elif i <= 6:
                exp_trajectories[2][9].append((float(i), 6 - i))
            elif i <= 9:
                exp_trajectories[2][9].append((float(i), i - 6))
            else:
                exp_trajectories[2][9].append((float(i), 12 - i))
        exp_trajectories[2][9].append(exp_trajectories[2][9][-1])

        profiles = self.analyzer.generate_time_dep_profiles()
        for i in range(3):
            self.assertDictsAlmostEqual(
                profiles["species_profiles"][i], exp_trajectories[i]
            )

        # Test intermediates analysis
        intermediates_analysis = self.analyzer.analyze_intermediates(
            profiles["species_profiles"]
        )
        exp_intermediates = {
            "h2o-": {
                "frequency": 2 / 3,
                "lifetime": (4.5, np.std([6, 3])),
                "t_max": (4.5, 1.5),
                "amt_produced": (6.0, 0.0),
                "amt_consumed": (6.0, 0.0),
            },
            "h2o": {
                "frequency": 1.0,
                "lifetime": (1.0, 0.0),
                "t_max": (0.0, 0.0),
                "amt_produced": (16 / 3, np.std([6, 4, 6])),
                "amt_consumed": (20 / 3, np.std([6, 8, 6])),
            },
        }
        exp_sorted_intermediates = sorted(
            [(ind, data) for ind, data in exp_intermediates.items()],
            key=lambda x: x[1]["amt_produced"][0],
            reverse=True,
        )
        self.assertCountEqual(intermediates_analysis, exp_sorted_intermediates)

        # Test final state analysis
        actual_final_states = self.analyzer.final_state_analysis(
            profiles["final_states"]
        )
        expected_final_states = dict()
        unchanged_species = ["h2", "o2", "h+", "oh-"]
        for id in unchanged_species:
            expected_final_states[id] = (
                self.num_mols,
                0,
            )  # (avg, std dev) of final state
        h2o_final_states = [self.num_mols, self.num_mols - 4, self.num_mols]
        expected_final_states["h2o"] = (
            np.mean(h2o_final_states),
            np.std(h2o_final_states),
        )
        h2ominus_final_states = [0, 4, 0]
        expected_final_states["h2o-"] = (
            np.mean(h2ominus_final_states),
            np.std(h2ominus_final_states),
        )
        expected_sorted_final_states = sorted(
            [
                (entry_id, data_tup)
                for entry_id, data_tup in expected_final_states.items()
            ],
            key=lambda x: x[1][0],
            reverse=True,
        )
        self.assertCountEqual(expected_sorted_final_states, actual_final_states)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_quantify_rank_reactions(self):
        reaction_analysis = self.analyzer.quantify_rank_reactions()
        expected = [
            (self.rxn_a, (20 / 3, np.std(np.array([6, 8, 6])))),
            (self.rxn_b, (16 / 3, np.std(np.array([6, 4, 6])))),
        ]
        self.assertCountEqual(reaction_analysis, expected)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_frequency_analysis(self):
        freq_analysis = self.analyzer.frequency_analysis(
            [self.rxn_a, self.rxn_b], [9, 10], 2
        )
        expected_rxn = {
            self.rxn_a: [
                (3, np.mean([1, 1, 1 / 2]), np.std([1, 1, 1 / 2])),
                (9, np.mean([0, 1 / 3, 1 / 2]), np.std([0, 1 / 3, 1 / 2])),
            ],
            self.rxn_b: [
                (3, np.mean([0, 0, 1 / 2]), np.std([0, 0, 1 / 2])),
                (9, np.mean([1, 2 / 3, 1 / 2]), np.std([1, 2 / 3, 1 / 2])),
            ],
        }
        expected_spec = {
            9: [
                (3, np.mean([1, 1, 1 / 2]), np.std([1, 1, 1 / 2])),
                (9, np.mean([0, 1 / 3, 1 / 2]), np.std([0, 1 / 3, 1 / 2])),
            ],
            10: [
                (3, np.mean([0, 0, 1 / 2]), np.std([0, 0, 1 / 2])),
                (9, np.mean([1, 2 / 3, 1 / 2]), np.std([1, 2 / 3, 1 / 2])),
            ],
        }
        self.assertDictsAlmostEqual(freq_analysis["reaction_data"], expected_rxn)
        self.assertDictsAlmostEqual(freq_analysis["species_data"], expected_spec)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_correlate_reactions(self):
        expected_correlation = {
            self.rxn_a: {
                "time": (1.0, 0.0),
                "steps": (1.0, 0.0),
                "occurrences": (np.mean([1, 1, 2]), np.std([1, 1, 2])),
            },
            self.rxn_b: {
                "time": (1.0, 0.0),
                "steps": (1.0, 0.0),
                "occurrences": (1 / 3, np.std([0, 0, 1])),
            },
        }
        rxn_correlations = self.analyzer.correlate_reactions([self.rxn_a, self.rxn_b])
        self.assertDictEqual(expected_correlation, rxn_correlations)


if __name__ == "__main__":
    unittest.main()
