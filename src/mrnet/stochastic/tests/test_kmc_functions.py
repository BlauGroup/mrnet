import os
import unittest
import copy

from pymatgen.util.testing import PymatgenTest
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry

from mrnet.stochastic.kmc import *
from mrnet.network.reaction_network import ReactionNetwork

try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

test_dir = os.path.join(os.path.dirname(__file__))


class TestKMCReactionPropagator(PymatgenTest):
    def setUp(self):
        """ Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O
        """
        self.volume = 10**-24  # m^3

        # 100 molecules each of H2O, H2, O2, OH-, H+
        self.num_mols = int(100)
        self.concentration = self.num_mols / N_A / self.volume / 1000

        # Make molecule objects
        H2O_mol = Molecule.from_file(os.path.join(test_dir, "H2O.xyz"))
        H2O_mol1 = copy.deepcopy(H2O_mol)
        H2O_mol_1 = copy.deepcopy(H2O_mol)
        H2O_mol1.set_charge_and_spin(charge=1)
        H2O_mol_1.set_charge_and_spin(charge=-1)

        H2_mol = Molecule.from_file(os.path.join(test_dir, "H2.xyz"))
        H2_mol1 = copy.deepcopy(H2_mol)
        H2_mol_1 = copy.deepcopy(H2_mol)
        H2_mol1.set_charge_and_spin(charge=1)
        H2_mol_1.set_charge_and_spin(charge=-1)

        O2_mol = Molecule.from_file(os.path.join(test_dir, "O2.xyz"))
        O2_mol1 = copy.deepcopy(O2_mol)
        O2_mol_1 = copy.deepcopy(O2_mol)
        O2_mol1.set_charge_and_spin(charge=1)
        O2_mol_1.set_charge_and_spin(charge=-1)

        OH_mol = Molecule.from_file(os.path.join(test_dir, "OH.xyz"))
        OH_mol1 = copy.deepcopy(OH_mol)
        OH_mol_1 = copy.deepcopy(OH_mol)
        OH_mol1.set_charge_and_spin(charge=1)
        OH_mol_1.set_charge_and_spin(charge=-1)

        H_mol = Molecule.from_file(os.path.join(test_dir, "H.xyz"))
        H_mol1 = copy.deepcopy(H_mol)
        H_mol_1 = copy.deepcopy(H_mol)
        H_mol1.set_charge_and_spin(charge=1)
        H_mol_1.set_charge_and_spin(charge=-1)

        O_mol = Molecule.from_file(os.path.join(test_dir, "O.xyz"))
        O_mol1 = copy.deepcopy(O_mol)
        O_mol_1 = copy.deepcopy(O_mol)
        O_mol1.set_charge_and_spin(charge=1)
        O_mol_1.set_charge_and_spin(charge=-1)

        # Make molecule entries
        # H2O 1-3
        if ob:
            H2O = MoleculeEntry(H2O_mol, energy=-76.4447861695239, correction=0, enthalpy=15.702, entropy=46.474,
                                parameters=None, entry_id='h2o', attribute=None)
            H2O_1 = MoleculeEntry(H2O_mol_1, energy=-76.4634569330715, correction=0, enthalpy=13.298, entropy=46.601,
                                  parameters=None, entry_id='h2o-', attribute=None)
            H2O_1p = MoleculeEntry(H2O_mol1, energy=-76.0924662469782, correction=0, enthalpy=13.697, entropy=46.765,
                                   parameters=None, entry_id='h2o+', attribute=None)
            # H2 4-6
            H2 = MoleculeEntry(H2_mol, energy=-1.17275734244991, correction=0, enthalpy=8.685, entropy=31.141,
                               parameters=None, entry_id='h2', attribute=None)
            H2_1 = MoleculeEntry(H2_mol_1, energy=-1.16232420718418, correction=0, enthalpy=3.56, entropy=33.346,
                                 parameters=None, entry_id='h2-', attribute=None)
            H2_1p = MoleculeEntry(H2_mol1, energy=-0.781383960574136, correction=0, enthalpy=5.773, entropy=32.507,
                                  parameters=None, entry_id='h2+', attribute=None)

            # OH 7-9
            OH = MoleculeEntry(OH_mol, energy=-75.7471080255785, correction=0, enthalpy=7.659, entropy=41.21,
                               parameters=None, entry_id='oh', attribute=None)
            OH_1 = MoleculeEntry(OH_mol_1, energy=-75.909589774742, correction=0, enthalpy=7.877, entropy=41.145,
                                 parameters=None, entry_id='oh-', attribute=None)
            OH_1p = MoleculeEntry(OH_mol1, energy=-75.2707068199185, correction=0, enthalpy=6.469, entropy=41.518,
                                  parameters=None, entry_id='oh+', attribute=None)
            # O2 10-12
            O2 = MoleculeEntry(O2_mol, energy=-150.291045922131, correction=0, enthalpy=4.821, entropy=46.76,
                               parameters=None, entry_id='o2', attribute=None)
            O2_1p = MoleculeEntry(O2_mol1, energy=-149.995474036502, correction=0, enthalpy=5.435, entropy=46.428,
                                  parameters=None, entry_id='o2+', attribute=None)
            O2_1 = MoleculeEntry(O2_mol_1, energy=-150.454499528454, correction=0, enthalpy=4.198, entropy=47.192,
                                 parameters=None, entry_id='o2-', attribute=None)

            # O 13-15
            O = MoleculeEntry(O_mol, energy=-74.9760564004, correction=0, enthalpy=1.481, entropy=34.254,
                              parameters=None, entry_id='o', attribute=None)
            O_1 = MoleculeEntry(O_mol_1, energy=-75.2301047938, correction=0, enthalpy=1.481, entropy=34.254,
                                parameters=None, entry_id='o-', attribute=None)
            O_1p = MoleculeEntry(O_mol1, energy=-74.5266804995, correction=0, enthalpy=1.481, entropy=34.254,
                                 parameters=None, entry_id='o+', attribute=None)
            # H 15-18
            H = MoleculeEntry(H_mol, energy=-0.5004488848, correction=0, enthalpy=1.481, entropy=26.014,
                              parameters=None, entry_id='h', attribute=None)
            H_1p = MoleculeEntry(H_mol1, energy=-0.2027210483, correction=0, enthalpy=1.481, entropy=26.066,
                                 parameters=None, entry_id='h+', attribute=None)
            H_1 = MoleculeEntry(H_mol_1, energy=-0.6430639079, correction=0, enthalpy=1.481, entropy=26.014,
                                parameters=None, entry_id='h-', attribute=None)

            self.mol_entries = [H2O, H2O_1, H2O_1p, H2, H2_1, H2_1p,
                                OH, OH_1, OH_1p, O2, O2_1p, O2_1,
                                O, O_1, O_1p, H, H_1p, H_1]

            self.reaction_network = ReactionNetwork.from_input_entries(self.mol_entries, electron_free_energy=-2.15)
            self.reaction_network.build()
            # print('number of reactions: ', len(self.reaction_network.reactions))
            # Only H2O, H2, O2 present initially
            self.initial_conditions = {'h2o': self.concentration, 'h2': self.concentration, 'o2': self.concentration,
                                       'oh-': self.concentration, 'h+': self.concentration}
            self.num_species = len(self.reaction_network.entries_list)
            self.num_reactions = len(self.reaction_network.reactions)

            self.reactants = np.ones((self.num_reactions, 2), dtype=int) * -1
            self.products = np.ones((self.num_reactions, 2), dtype=int) * -1
            self.rate_constants = np.zeros(2*self.num_reactions)
            self.coord_array = np.zeros(2*self.num_reactions)

            self.initial_state = [0 for i in range(self.num_species)]
            self.initial_state[2] = self.num_mols  # h+
            self.initial_state[3] = self.num_mols  # oh-
            self.initial_state[7] = self.num_mols  # h2
            self.initial_state[10] = self.num_mols  # h2o
            self.initial_state[16] = self.num_mols  # o2

            self.molid_ind_mapping = dict()
            species_rxn_mapping_list = [[] for i in range(self.num_species)]

            for ind, spec in enumerate(self.reaction_network.entries_list):
                self.molid_ind_mapping[spec.entry_id] = ind
            # print('molid_ind mapping:', self.molid_ind_mapping)
            # construct the product and reactants arrays
            for ind, reaction in enumerate(self.reaction_network.reactions):
                num_reactants_for = list()
                num_reactants_rev = list()
                for idx, react in enumerate(reaction.reactants):
                    mol_ind = self.molid_ind_mapping[react.entry_id]
                    self.reactants[ind, idx] = mol_ind
                    num_reactants_for.append(self.initial_state[mol_ind])
                    species_rxn_mapping_list[mol_ind].append(2*ind)
                for idx, prod in enumerate(reaction.products):
                    mol_ind = self.molid_ind_mapping[prod.entry_id]
                    self.products[ind, idx] = mol_ind
                    num_reactants_rev.append(self.initial_state[mol_ind])
                    species_rxn_mapping_list[mol_ind].append(2*ind+1)

                self.rate_constants[2*ind] = reaction.rate_constant()["k_A"]
                self.rate_constants[2*ind+1] = reaction.rate_constant()["k_B"]

                # set up coordination array
                if len(reaction.reactants) == 1:
                    self.coord_array[2 * ind] = num_reactants_for[0]
                elif (len(reaction.reactants) == 2) and (reaction.reactants[0] == reaction.reactants[1]):
                    self.coord_array[2 * ind] = num_reactants_for[0] * (num_reactants_for[0] - 1)
                elif (len(reaction.reactants) == 2) and (reaction.reactants[0] != reaction.reactants[1]):
                    self.coord_array[2 * ind] = num_reactants_for[0] * num_reactants_for[1]
                else:
                    raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
                # For reverse reaction
                if len(reaction.products) == 1:
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0]
                elif (len(reaction.products) == 2) and (reaction.products[0] == reaction.products[1]):
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
                elif (len(reaction.products) == 2) and (reaction.products[0] != reaction.products[1]):
                    self.coord_array[2 * ind + 1] = num_reactants_rev[0] * num_reactants_rev[1]
                else:
                    raise RuntimeError("Only single and bimolecular reactions supported by this simulation")

            self.propensities = np.multiply(self.coord_array, self.rate_constants)

            # Set up molind_rxn_mapping
            spec_rxn_map_lengths = [len(rxn_list) for rxn_list in species_rxn_mapping_list]
            max_map_length = max(spec_rxn_map_lengths)
            self.species_rxn_mapping = np.ones((self.num_species, max_map_length), dtype=int) * -1
            for ind, rxn_list in enumerate(species_rxn_mapping_list):
                if len(rxn_list) == max_map_length:
                    self.species_rxn_mapping[ind, :] = rxn_list
                else:
                    self.species_rxn_mapping[ind, : len(rxn_list)] = rxn_list

    def tearDown(self) -> None:
        if ob:
            del self.volume
            del self.num_mols
            del self.concentration
            del self.mol_entries
            del self.reaction_network
            del self.initial_conditions
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

        [initial_state, initial_state_dict, species_rxn_mapping, reactant_array, product_array, coord_array,
         rate_constants, propensities, molid_index_mapping] = \
            initialize_simulation(self.reaction_network, self.initial_conditions, self.volume)

        # Verify initial state
        num_species = len(self.reaction_network.entries_list)
        exp_initial_state = [0 for i in range(num_species)]
        exp_initial_state[2] = self.num_mols  # h+
        exp_initial_state[3] = self.num_mols  # oh-
        exp_initial_state[7] = self.num_mols  # h2
        exp_initial_state[10] = self.num_mols  #h2o
        exp_initial_state[16] = self.num_mols  #o2

        # exp_species_rxn_mapping = -1 * np.ones((18, 16))
        # exp_species_rxn_mapping_list = list()
        # exp_species_rxn_mapping_list.append([0, 13, 16, 19, 20, 21, 23, 27, 29, 32, 35]) # Reactions for H-
        # exp_species_rxn_mapping_list.append([0, 1, 12, 15, 18, 19, 20, 22, 24,25, 26, 28, 31, 34, 37, 39]) #H
        # exp_species_rxn_mapping_list.append([1, 14, 17, 21, 23, 24, 25, 30, 33, 36, 38]) #H+
        # exp_species_rxn_mapping_list.append([2, 12, 13, 26, 28, 30, 33]) #oh-
        # exp_species_rxn_mapping_list.append([2, 3, 14, 15, 16, 27, 29, 31, 34, 36, 38]) #oh
        # exp_species_rxn_mapping_list.append([3, 17, 18, 32, 35, 37, 39]) #oh+
        # exp_species_rxn_mapping_list.append([4, 19, 20]) #h2-
        # exp_species_rxn_mapping_list.append([4, 5, 21, 22, 23]) #h2
        # exp_species_rxn_mapping_list.append([5, 24, 25]) #h2+
        # exp_species_rxn_mapping_list.append([6, 26, 27, 28, 29]) #h2o-
        # exp_species_rxn_mapping_list.append([6, 7, 30, 31, 32, 33, 34, 35]) #h2o
        # exp_species_rxn_mapping_list.append([7, 36, 37, 38, 39]) #h2o+
        # exp_species_rxn_mapping_list.append([8, 14, 40, 41, 42, 44]) #o-
        # exp_species_rxn_mapping_list.append([8, 9, 13, 15, 17, 40, 41, 43, 45, 46]) #o
        # exp_species_rxn_mapping_list.append([9, 16, 18, 42, 44, 45, 46]) #o+
        # exp_species_rxn_mapping_list.append([10, 40, 41]) #o2-
        # exp_species_rxn_mapping_list.append([10, 11, 42, 43, 44]) #o2
        # exp_species_rxn_mapping_list.append([11, 45, 46]) #o2+

        self.assertEqual(exp_initial_state, initial_state)
        self.assertEqual(self.initial_state, initial_state)
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
            if ((['h2'] == reactants) and (['h2+'] == products)) or ((['h2o'] == reactants) and (['h2o-'] == products))\
                    or ((['h2'] == reactants) and (['h-', 'h+'] == products)):
                reactions_sequence.extend([2*ind, 2*ind + 1, 2*ind])
            elif ((['h2+'] == reactants) and (['h2'] == products)) or \
                    ((['h2o-'] == reactants) and (['h2o'] == products)) or \
                    ((['h-', 'h+'] == reactants) and (['h'] == products)):
                reactions_sequence.extend([2*ind+1, 2*ind, 2*ind+1])

        num_iterations = 10
        state = np.array(self.initial_state)

        for i in range(num_iterations):
            for rxn_ind in reactions_sequence:
                if rxn_ind % 2:
                    reverse = True
                else:
                    reverse = False
                converted_rxn_ind = math.floor(rxn_ind/2)
                state = update_state(self.reactants, self.products, state, converted_rxn_ind, reverse)

        expected_state = [0 for i in range(self.num_species)]
        expected_state[self.molid_ind_mapping['h-']] = num_iterations
        expected_state[self.molid_ind_mapping['h+']] = self.num_mols + num_iterations
        expected_state[self.molid_ind_mapping['oh-']] = self.num_mols
        expected_state[self.molid_ind_mapping['h2']] = self.num_mols - 2*num_iterations
        expected_state[self.molid_ind_mapping['h2+']] = num_iterations
        expected_state[self.molid_ind_mapping['h2o-']] = num_iterations
        expected_state[self.molid_ind_mapping['h2o']] = self.num_mols - num_iterations
        expected_state[self.molid_ind_mapping['o2']] = self.num_mols
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
            if (['h2'] == reactants) and (['h2+'] == products):
                run_test = 1
                reaction_sequence = [2*ind, 2*ind+1]
            elif (['h2+'] == reactants) and (['h2'] == products):
                run_test = 1
                reaction_sequence = [2*ind+1, 2*ind]
            elif (['h2o'] == reactants) and (['oh-', 'h+'] == products):
                run_test = 2
                reaction_sequence = [2 * ind, 2 * ind + 1]
            elif (['oh-', 'h+'] == reactants) and (['h2o'] == products):
                run_test = 2
                reaction_sequence = [2*ind+1, 2*ind]

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
                    coords.append(get_coordination(self.reactants, self.products, state, converted_rxn_ind, reverse))

                self.assertEqual(expected_coords, coords)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_kmc_simulate(self):
        t_steps = 1
        iterations = 10000
        reaction_history = list()
        time_steps = list()
        reaction_frequency = [0 for i in range(2*self.num_reactions)]
        total_propensity = np.sum(self.propensities)
        exp_tau = 1 / total_propensity  # expectation value of first time step
        rxn_probability = self.propensities / total_propensity  # expected frequencies of each reaction
        # for rxn in self.reaction_network.reactions:
        for i in range(iterations):
            # run simulation with initial conditions, 1 time step
            sim_data = kmc_simulate(t_steps, self.coord_array, self.rate_constants, self.propensities,
                                    self.species_rxn_mapping, self.reactants, self.products,
                                    np.array(self.initial_state))
            if (t_steps != len(sim_data[0])) or (t_steps != len(sim_data[1])):
                raise RuntimeError("There are more than the specified time steps for this simulation.")
            reaction_history.append(int(sim_data[0][0]))
            time_steps.append(sim_data[1][0])

        reaction_history = np.array(reaction_history)
        time_steps = np.array(time_steps)
        avg_tau = np.average(time_steps)
        self.assertAlmostEqual(avg_tau, exp_tau)


if __name__ == "__main__":
    unittest.main()