# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A
from numba import jit
import copy

__author__ = "Ronald Kam, Evan Spotte-Smith, Xiaowei Xie"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

"""
Kinetic Monte Carlo (kMC) simulation for a reaction network, assuming spatial homogeneity. Simulation can be performed
with and without ReactionNetwork objects. The version without ReactionNetwork objects is computationally cheaper.
The algorithm is described by Gillespie (1976).

"""


def initialize_simulation(reaction_network, initial_cond, volume=10 ** -24):
    """
    Initial loop through reactions to create lists, mappings, and initial states needed for simulation without
    reaction network objects.

    Args:
        reaction_network: Fully generated ReactionNetwork
        initial_cond: dict mapping mol_index to initial concentration [M]. mol_index is entry position in
                             reaction_network.entries_list
        volume: float of system volume

    :return:
        initial_state: array of initial molecule amounts, indexed corresponding to reaction_network.entries_list
        initial_state_dict: dict mapping molecule index to initial molecule amounts
        species_rxn_mapping: 2d array; each row i contains reactions which molecule_i takes part in
        molid_index_mapping: mapping between species entry id and its molecule index
        reactants_array: (n_rxns x 2) array, each row containing reactant mol_index of forward reaction
        products_array: (n_rxns x 2) array, each row containing product mol_index of forward reaction
        coord_array: (2*n_rxns x 1) array, with coordination number of each for and rev rxn: [c1_f, c1_r, c2_f, c2_r...]
        rate_constants: (2*n_rxns x 1) array, with rate constant of each for and rev rxn: [k1_f, k1_r, k2_f, k2_r ...]
        propensities: (2*n_rxns x 1) array of reaction propensities, defined as coord_num*rate_constant

    """

    num_rxns = len(reaction_network.reactions)
    num_species = len(reaction_network.entries_list)
    molid_index_mapping = dict()
    initial_state = [0 for i in range(num_species)]
    initial_state_dict = dict()

    for ind, mol in enumerate(reaction_network.entries_list):
        molid_index_mapping[mol.entry_id] = ind
        this_c = initial_cond.get(mol.entry_id, 0)
        this_mol_amt = int(volume * N_A * 1000 * this_c)
        initial_state[ind] = this_mol_amt
        if mol.entry_id in initial_cond:
            initial_state_dict[ind] = this_mol_amt

    # Initially compile each species' reactions in lists, later convert to a 2d array
    species_rxn_mapping_list = [[] for j in range(num_species)]
    reactant_array = -1 * np.ones((num_rxns, 2), dtype=int)
    product_array = -1 * np.ones((num_rxns, 2), dtype=int)
    coord_array = np.zeros(2 * num_rxns)
    rate_constants = np.zeros(2 * num_rxns)
    for id, reaction in enumerate(reaction_network.reactions):
        # Keep track of reactant amounts, for later calculating coordination number
        num_reactants_for = list()
        num_reactants_rev = list()
        rate_constants[2 * id] = reaction.k_A
        rate_constants[2 * id + 1] = reaction.k_B
        for idx, react in enumerate(reaction.reactants):
            # for each reactant, need to find the corresponding mol_id with the index
            mol_ind = molid_index_mapping[react.entry_id]
            reactant_array[id, idx] = mol_ind
            species_rxn_mapping_list[mol_ind].append(2 * id)
            num_reactants_for.append(initial_state[mol_ind])

        for idx, prod in enumerate(reaction.products):
            mol_ind = molid_index_mapping[prod.entry_id]
            product_array[id, idx] = mol_ind
            species_rxn_mapping_list[mol_ind].append(2 * id + 1)
            num_reactants_rev.append(initial_state[mol_ind])

        if len(reaction.reactants) == 1:
            coord_array[2 * id] = num_reactants_for[0]
        elif (len(reaction.reactants) == 2) and (
            reaction.reactants[0] == reaction.reactants[1]
        ):
            coord_array[2 * id] = num_reactants_for[0] * (num_reactants_for[0] - 1)
        elif (len(reaction.reactants) == 2) and (
            reaction.reactants[0] != reaction.reactants[1]
        ):
            coord_array[2 * id] = num_reactants_for[0] * num_reactants_for[1]
        else:
            raise RuntimeError(
                "Only single and bimolecular reactions supported by this simulation"
            )
        # For reverse reaction
        if len(reaction.products) == 1:
            coord_array[2 * id + 1] = num_reactants_rev[0]
        elif (len(reaction.products) == 2) and (
            reaction.products[0] == reaction.products[1]
        ):
            coord_array[2 * id + 1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
        elif (len(reaction.products) == 2) and (
            reaction.products[0] != reaction.products[1]
        ):
            coord_array[2 * id + 1] = num_reactants_rev[0] * num_reactants_rev[1]
        else:
            raise RuntimeError(
                "Only single and bimolecular reactions supported by this simulation"
            )
    rxn_mapping_lengths = [len(rxn_list) for rxn_list in species_rxn_mapping_list]
    max_mapping_length = max(rxn_mapping_lengths)
    species_rxn_mapping = -1 * np.ones((num_species, max_mapping_length), dtype=int)

    for index, rxn_list in enumerate(species_rxn_mapping_list):
        this_map_length = rxn_mapping_lengths[index]
        if this_map_length == max_mapping_length:
            species_rxn_mapping[index, :] = rxn_list
        else:
            species_rxn_mapping[
                index, : this_map_length - max_mapping_length
            ] = rxn_list
    propensities = np.multiply(coord_array, rate_constants)
    return [
        np.array(initial_state, dtype=int),
        initial_state_dict,
        species_rxn_mapping,
        reactant_array,
        product_array,
        coord_array,
        rate_constants,
        propensities,
        molid_index_mapping,
    ]


@jit(nopython=True, parallel=True)
def kmc_simulate(
    time_steps,
    coord_array,
    rate_constants,
    propensity_array,
    species_rxn_mapping,
    reactants,
    products,
    state,
):
    """
    KMC Simulation of reaction network and specified initial conditions. Args are all Numpy arrays, to allow
    computational speed up with Numba.

    Args:
        time_steps: int number of time steps desired to run
        coord_array: array containing coordination numbers of for and rev rxns.
        rate_constants: array containing rate constants of for and rev rxns
        propensity_array: array containing propensities of for and rev rxns
        species_rxn_mapping: 2d array; each row i contains reactions which molecule_i takes part in
        reactants: (n_rxns x 2) array, each row containing reactant mol_index of forward reaction
        products: (n_rxns x 2) array, each row containing product mol_index of forward reaction
        state: array of initial molecule amounts, indexed corresponding to reaction_network.entries_list

    :return: A (2 x time_steps) Numpy array. First row contains the indeces of reactions that occurred.
             Second row are the time steps generated at each iteration.

    """

    total_propensity = np.sum(propensity_array)
    reaction_history = [0 for step in range(time_steps)]
    times = [0.0 for step in range(time_steps)]
    relevant_ind = np.where(propensity_array > 0)[
        0
    ]  # Take advantage of sparsity - many propensities will be 0.
    for step_counter in range(time_steps):
        r1 = random.random()
        r2 = random.random()
        tau = -np.log(r1) / total_propensity
        random_propensity = r2 * total_propensity
        abrgd_reaction_choice_ind = np.where(
            np.cumsum(propensity_array[relevant_ind]) >= random_propensity
        )[0][0]
        reaction_choice_ind = relevant_ind[abrgd_reaction_choice_ind]
        converted_rxn_ind = math.floor(reaction_choice_ind / 2)
        if reaction_choice_ind % 2:
            reverse = True
        else:
            reverse = False

        state = update_state(reactants, products, state, converted_rxn_ind, reverse)
        # Log the reactions that need to be altered after reaction is performed, for the coordination array
        reactions_to_change = list()
        for reactant_id in reactants[converted_rxn_ind, :]:
            if reactant_id == -1:
                continue
            else:
                reactions_to_change.extend(list(species_rxn_mapping[reactant_id, :]))
        for product_id in products[converted_rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                reactions_to_change.extend(list(species_rxn_mapping[product_id, :]))
        rxns_change = set(reactions_to_change)
        for rxn_ind in rxns_change:
            if rxn_ind == -1:
                continue
            elif rxn_ind % 2:
                this_reverse = True
            else:
                this_reverse = False
            this_h = get_coordination(
                reactants, products, state, math.floor(rxn_ind / 2), this_reverse
            )
            coord_array[rxn_ind] = this_h

        propensity_array = np.multiply(rate_constants, coord_array)
        relevant_ind = np.where(propensity_array > 0)[0]
        total_propensity = np.sum(propensity_array[relevant_ind])
        reaction_history[step_counter] = int(reaction_choice_ind)
        times[step_counter] = tau

    return np.vstack((np.array(reaction_history), np.array(times)))


@jit(nopython=True)
def update_state(reactants, products, state, rxn_ind, reverse):
    """
    Updating the system state based on chosen reaction, during kMC simulation.

    Args:
        reactants: (n_rxns x 2) array, each row containing reactant mol_index of forward reaction
        products: (n_rxns x 2) array, each row containing product mol_index of forward reaction
        state: array of initial molecule amounts, indexed corresponding to reaction_network.entries_list
        rxn_ind: int of reaction index, corresponding to position in reaction_network.reactions list
        reverse: bool of whether this is the reverse reaction or not

    :return: updated state array, after performing the specified reaction
    """

    if rxn_ind == -1:
        raise RuntimeError("Incorrect reaction index when updating state")
    if reverse:
        for reactant_id in products[rxn_ind, :]:
            if reactant_id == -1:
                continue
            else:
                state[reactant_id] -= 1
                if state[reactant_id] < 0:
                    raise ValueError("State invalid! Negative specie encountered")
        for product_id in reactants[rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                state[product_id] += 1
    else:
        for reactant_id in reactants[rxn_ind, :]:
            if reactant_id == -1:
                continue
            else:
                state[reactant_id] -= 1
                if state[reactant_id] < 0:
                    raise ValueError("State invalid! Negative specie encountered")
        for product_id in products[rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                state[product_id] += 1

    return state


@jit(nopython=True)
def get_coordination(reactants, products, state, rxn_id, reverse):
    """
    Calculate the coordination number of a reaction, for reactions involving two reactions of less.
    They are defined as follows:
    A -> B;  coord = n(A)
    A + A --> B; coord = n(A) * (n(A) - 1)
    A + B --> C; coord = n(A) * n(B)

    Args:
        reactants: (n_rxns x 2) array, each row containing reactant mol_index of forward reaction
        products: (n_rxns x 2) array, each row containing product mol_index of forward reaction
        state: array of initial molecule amounts, indexed corresponding to reaction_network.entries_list
        rxn_ind: int of reaction index, corresponding to position in reaction_network.reactions list
        reverse: bool of whether this is the reverse reaction or not

    :return: float of reaction coordination number
    """

    if reverse:
        reactant_array = products[rxn_id, :]
        num_reactants = len(np.where(reactant_array != -1)[0])
    else:
        reactant_array = reactants[rxn_id, :]
        num_reactants = len(np.where(reactant_array != -1)[0])

    num_mols_list = list()
    for reactant_id in reactant_array:
        num_mols_list.append(state[reactant_id])

    if num_reactants == 1:
        h_prop = num_mols_list[0]
    elif (num_reactants == 2) and (reactant_array[0] == reactant_array[1]):
        h_prop = num_mols_list[0] * (num_mols_list[0] - 1) / 2
    elif (num_reactants == 2) and (reactant_array[0] != reactant_array[1]):
        h_prop = num_mols_list[0] * num_mols_list[1]
    else:
        raise RuntimeError(
            "Only single and bimolecular reactions supported by this simulation"
        )

    return h_prop


class KmcDataAnalyzer:
    """
    Functions to analyze (function-based) KMC outputs from many simulation runs. Ideally, the reaction history and
    time history data are list of arrays.

    Args:
        reaction_network: fully generated ReactionNetwork, used for kMC simulation
        molid_ind_mapping: dict mapping each entry's id to its index; of form {entry_id: mol_index, ... }
        species_rxn_mapping: 2d array; each row i contains reactions which molecule_i takes part in
        initial_state_dict: dict mapping mol_id to its initial amount {mol1_id: amt_1, mol2_id: amt2 ... }
        products: (n_rxns x 2) array, each row containing product mol_index of forward reaction
        reactants: (n_rxns x 2) array, each row containing reactant mol_index of forward reaction
        reaction_history: list of arrays of reaction histories of each simulation.
        time_history: list of arrays of time histories of each simulation.

    """

    def __init__(
        self,
        reaction_network,
        molid_ind_mapping,
        species_rxn_mapping,
        initial_state_dict,
        products,
        reactants,
        reaction_history,
        time_history,
    ):
        self.reaction_network = reaction_network
        self.molid_ind_mapping = molid_ind_mapping
        self.species_rxn_mapping = species_rxn_mapping
        self.initial_state_dict = initial_state_dict
        self.products = products
        self.reactants = reactants
        self.reaction_history = reaction_history
        self.time_history = time_history
        self.num_sims = len(self.reaction_history)
        if self.num_sims != len(self.time_history):
            raise RuntimeError(
                "Number of datasets for rxn history and time step history should be same!"
            )
        self.molind_id_mapping = [
            mol.entry_id for mol in self.reaction_network.entries_list
        ]

    def generate_time_dep_profiles(self):
        """
        Generate plottable time-dependent profiles of species and rxns from raw KMC output, obtain final states.

        :return dict containing species profiles, reaction profiles, and final states from each simulation.
                {species_profiles: [ {mol_ind1: [(t0, n(t0)), (t1, n(t1)...], mol_ind2: [...] ,  ... }, {...}, ... ]
                reaction_profiles: [ {rxn_ind1: [t0, t1, ...], rxn_ind2: ..., ...}, {...}, ...]
                final_states: [ {mol_ind1: n1, mol_ind2: ..., ...}, {...}, ...] }

        """
        species_profiles = list()
        reaction_profiles = list()
        final_states = list()

        for n_sim in range(self.num_sims):
            sim_time_history = self.time_history[n_sim]
            sim_rxn_history = self.reaction_history[n_sim]
            sim_species_profile = dict()
            sim_rxn_profile = dict()
            cumulative_time = list(np.cumsum(np.array(sim_time_history)))
            state = copy.deepcopy(self.initial_state_dict)
            for mol_ind in state:
                sim_species_profile[mol_ind] = [(0.0, self.initial_state_dict[mol_ind])]
            total_iterations = len(sim_rxn_history)

            for iter in range(total_iterations):
                rxn_ind = sim_rxn_history[iter]
                t = cumulative_time[iter]
                if rxn_ind not in sim_rxn_profile:
                    sim_rxn_profile[rxn_ind] = [t]
                else:
                    sim_rxn_profile[rxn_ind].append(t)
                converted_ind = math.floor(rxn_ind / 2)

                if rxn_ind % 2:
                    reacts = self.products[converted_ind, :]
                    prods = self.reactants[converted_ind, :]
                else:
                    reacts = self.reactants[converted_ind, :]
                    prods = self.products[converted_ind, :]

                for r_ind in reacts:
                    if r_ind == -1:
                        continue
                    else:
                        try:
                            state[r_ind] -= 1
                            if state[r_ind] < 0:
                                raise ValueError(
                                    "State invalid: negative specie: {}".format(r_ind)
                                )
                            sim_species_profile[r_ind].append((t, state[r_ind]))
                        except KeyError:
                            raise ValueError(
                                "Reactant specie {} given is not in state!".format(
                                    r_ind
                                )
                            )
                for p_ind in prods:
                    if p_ind == -1:
                        continue
                    else:
                        if (p_ind in state) and (p_ind in sim_species_profile):
                            state[p_ind] += 1
                            sim_species_profile[p_ind].append((t, state[p_ind]))
                        else:
                            state[p_ind] = 1
                            sim_species_profile[p_ind] = [(0.0, 0), (t, state[p_ind])]

            # for plotting convenience, add data point at final time
            for mol_ind in sim_species_profile:
                sim_species_profile[mol_ind].append(
                    (cumulative_time[-1], state[mol_ind])
                )

            species_profiles.append(sim_species_profile)
            reaction_profiles.append(sim_rxn_profile)
            final_states.append(state)

        return {
            "species_profiles": species_profiles,
            "reaction_profiles": reaction_profiles,
            "final_states": final_states,
        }

    def final_state_analysis(self, final_states):
        """
        Gather statistical analysis of the final states of simulation.

        Args:
            final_states: list of dicts of final states, as generated in generate_time_dep_profiles()

        :return: list of tuples containing statistical data for each species, sorted from highest to low avg occurrence
        """
        state_arrays = (
            dict()
        )  # For each molecule, compile an array of its final amounts
        for iter, final_state in enumerate(final_states):
            for mol_ind, amt in final_state.items():
                # Store the amount, and convert key from mol_ind to entry_id
                if self.molind_id_mapping[mol_ind] not in state_arrays:
                    state_arrays[self.molind_id_mapping[mol_ind]] = np.zeros(
                        self.num_sims
                    )
                state_arrays[self.molind_id_mapping[mol_ind]][iter] = amt
        analyzed_states = dict()  # will contain statistical results of final states
        for mol_entry, state_array in state_arrays.items():
            analyzed_states[mol_entry] = (np.mean(state_array), np.std(state_array))
        # Sort from highest avg final amount to lowest
        sorted_analyzed_states = sorted(
            [(entry_id, data_tup) for entry_id, data_tup in analyzed_states.items()],
            key=lambda x: x[1][0],
            reverse=True,
        )
        return sorted_analyzed_states

    def plot_species_profiles(
        self,
        species_profiles,
        final_states,
        num_label=12,
        num_plots=None,
        filename=None,
        file_dir=None,
    ):
        """
        Sorting and plotting species profiles for a specified number of simulations. The profiles might be very similar,
        so may not need to plot all of the runs for good understanding of results.

        Args:
            species_profiles: list of dicts of species as function of time, for each simulation
            final_states: list of dicts of final states of each simulation
            num_label: integer number of species in the legend
            filename (str)
            file_dir (str)

        """
        if num_plots is None:
            num_plots = self.num_sims
        elif num_plots > self.num_sims:
            num_plots = self.num_sims

        for n_sim in range(num_plots):
            # Sorting and plotting:
            fig, ax = plt.subplots()
            sorted_state = sorted(
                [(k, v) for k, v in final_states[n_sim].items()],
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_inds = [mol_tuple[0] for mol_tuple in sorted_state]
            sorted_ind_id_mapping = dict()
            iter_counter = 0
            for id, ind in self.molid_ind_mapping.items():
                if ind in sorted_inds[:num_label]:
                    sorted_ind_id_mapping[ind] = id
                    iter_counter += 1
                if iter_counter == num_label:
                    break

            colors = plt.cm.get_cmap("hsv", num_label)
            this_id = 0
            t_end = sum(self.time_history[n_sim])
            for mol_ind in species_profiles[n_sim]:
                # ts = np.append(np.array([e[0] for e in species_profiles[n_sim][mol_ind]]), t_end)
                ts = np.array([e[0] for e in species_profiles[n_sim][mol_ind]])
                nums = np.array([e[1] for e in species_profiles[n_sim][mol_ind]])

                if mol_ind in sorted_inds[:num_label]:
                    mol_id = sorted_ind_id_mapping[mol_ind]
                    for entry in self.reaction_network.entries_list:
                        if mol_id == entry.entry_id:
                            this_composition = (
                                entry.molecule.composition.alphabetical_formula
                            )
                            this_charge = entry.molecule.charge
                            this_label = this_composition + " " + str(this_charge)
                            this_color = colors(this_id)
                            this_id += 1
                            break
                    ax.plot(ts, nums, label=this_label, color=this_color)
                else:
                    ax.plot(ts, nums)

            title = "KMC simulation, total time {}".format(t_end)
            ax.set(title=title, xlabel="Time (s)", ylabel="# Molecules")
            ax.legend(
                loc="upper right", bbox_to_anchor=(1, 1), ncol=2, fontsize="small"
            )

            sim_filename = filename + "_run_" + str(n_sim + 1)
            if file_dir is None:
                plt.show()
            else:
                plt.savefig(file_dir + "/" + sim_filename)

    def analyze_intermediates(self, species_profiles, cutoff=0.9):
        """
        Identify intermediates from species vs time profiles. Species are intermediates if consumed nearly as much
        as they are created.

        Args:
            species_profile: Dict of list of tuples, as generated in generate_time_dep_profiles()
            cutoff: (float) fraction to adjust definition of intermediate

        :return: Analyzed data in a dict, of the form:
            {mol1: {'freqency': (float), 'lifetime': (avg, std), 't_max': (avg, std), 'amt_produced': (avg, std)},
            mol2: {...}, ... }
        """
        intermediates = dict()
        for n_sim in range(self.num_sims):
            for mol_ind, prof in species_profiles[n_sim].items():
                history = np.array([t[1] for t in prof])
                diff_history = np.diff(history)
                max_amt = max(history)
                amt_produced = np.sum(diff_history == 1)
                amt_consumed = np.sum(diff_history == -1)
                # Identify the intermediate, accounting for fluctuations
                if (amt_produced >= 3) and (amt_consumed > amt_produced * cutoff):
                    if mol_ind not in intermediates:
                        intermediates[mol_ind] = dict()
                        intermediates[mol_ind]["lifetime"] = list()
                        intermediates[mol_ind]["amt_produced"] = list()
                        intermediates[mol_ind]["t_max"] = list()
                        intermediates[mol_ind]["amt_consumed"] = list()
                    # Intermediate lifetime is approximately the time from its max amount to when nearly all consumed
                    max_ind = np.where(history == max_amt)[0][0]
                    t_max = prof[max_ind][0]
                    for state in prof[max_ind + 1 :]:
                        if state[1] < (1 - cutoff) * amt_produced + history[0]:
                            intermediates[mol_ind]["lifetime"].append(state[0] - t_max)
                            intermediates[mol_ind]["t_max"].append(t_max)
                            intermediates[mol_ind]["amt_produced"].append(amt_produced)
                            intermediates[mol_ind]["amt_consumed"].append(amt_consumed)
                            break

        intermediates_analysis = dict()
        for mol_ind in intermediates:
            entry_id = self.molind_id_mapping[mol_ind]
            intermediates_analysis[entry_id] = dict()  # convert keys to entry id
            if len(intermediates[mol_ind]["lifetime"]) != len(
                intermediates[mol_ind]["t_max"]
            ):
                raise RuntimeError("Intermediates data should be of the same length")
            intermediates_analysis[entry_id]["frequency"] = (
                len(intermediates[mol_ind]["lifetime"]) / self.num_sims
            )
            lifetime_array = np.array(intermediates[mol_ind]["lifetime"])
            intermediates_analysis[entry_id]["lifetime"] = (
                np.mean(lifetime_array),
                np.std(lifetime_array),
            )
            t_max_array = np.array(intermediates[mol_ind]["t_max"])
            intermediates_analysis[entry_id]["t_max"] = (
                np.mean(t_max_array),
                np.std(t_max_array),
            )
            amt_produced_array = np.array(intermediates[mol_ind]["amt_produced"])
            intermediates_analysis[entry_id]["amt_produced"] = (
                np.mean(amt_produced_array),
                np.std(amt_produced_array),
            )
            amt_consumed_array = np.array(intermediates[mol_ind]["amt_consumed"])
            intermediates_analysis[entry_id]["amt_consumed"] = (
                np.mean(amt_consumed_array),
                np.std(amt_produced_array),
            )

        # Sort by highest average amount produced
        sorted_intermediates_analysis = sorted(
            [
                (entry_id, mol_data)
                for entry_id, mol_data in intermediates_analysis.items()
            ],
            key=lambda x: x[1]["amt_produced"][0],
            reverse=True,
        )

        return sorted_intermediates_analysis

    def correlate_reactions(self, reaction_inds):
        """
        Correlate two reactions, by finding the average time and steps elapsed for rxn2 to fire after rxn1,
        and vice-versa.

        Args:
            reaction_inds: list, array, or tuple of two reaction indexes

        :return: dict containing analysis of how reactions are correlated {rxn1: {'time': (float), 'steps': (float),
                'occurrences': float}, rxn2: {...} }
        """
        correlation_data = dict()
        correlation_analysis = dict()
        for rxn_ind in reaction_inds:
            correlation_data[rxn_ind] = dict()
            correlation_data[rxn_ind]["time"] = list()
            correlation_data[rxn_ind]["steps"] = list()
            correlation_data[rxn_ind]["occurrences"] = list()
            correlation_analysis[rxn_ind] = dict()

        for n_sim in range(self.num_sims):
            cum_time = np.cumsum(self.time_history[n_sim])
            rxn_locations = dict()
            # Find the step numbers when reactions fire in the simulation
            for rxn_ind in reaction_inds:
                rxn_locations[rxn_ind] = list(
                    np.where(self.reaction_history[n_sim] == rxn_ind)[0]
                )
                rxn_locations[rxn_ind].append(len(self.reaction_history[n_sim]))
            # Correlate between each reaction
            for (rxn_ind, location_list) in rxn_locations.items():
                time_elapse = list()
                step_elapse = list()
                occurrences = 0
                for (rxn_ind_j, location_list_j) in rxn_locations.items():
                    if rxn_ind == rxn_ind_j:
                        continue
                    for i in range(1, len(location_list)):
                        for loc_j in location_list_j:

                            # Find location where reaction j happens after reaction i, before reaction i fires again
                            if (loc_j > location_list[i - 1]) and (
                                loc_j < location_list[i]
                            ):
                                time_elapse.append(
                                    cum_time[loc_j] - cum_time[location_list[i - 1]]
                                )
                                step_elapse.append(loc_j - location_list[i - 1])
                                occurrences += 1
                                break

                if len(time_elapse) == 0:
                    correlation_data[rxn_ind]["occurrences"].append(0)
                else:
                    correlation_data[rxn_ind]["time"].append(
                        np.mean(np.array(time_elapse))
                    )
                    correlation_data[rxn_ind]["steps"].append(
                        np.mean(np.array(step_elapse))
                    )
                    correlation_data[rxn_ind]["occurrences"].append(occurrences)

        for rxn_ind, data_dict in correlation_data.items():
            if len(data_dict["time"]) != 0:
                correlation_analysis[rxn_ind]["time"] = (
                    np.mean(np.array(data_dict["time"])),
                    np.std(np.array(data_dict["time"])),
                )
                correlation_analysis[rxn_ind]["steps"] = (
                    np.mean(np.array(data_dict["steps"])),
                    np.std(np.array(data_dict["steps"])),
                )
                correlation_analysis[rxn_ind]["occurrences"] = (
                    np.mean(np.array(data_dict["occurrences"])),
                    np.std(np.array(data_dict["occurrences"])),
                )
            else:
                print(
                    "Reaction ",
                    rxn_ind,
                    "does not lead to the other reaction in simulation ",
                    n_sim,
                )

        return correlation_analysis

    def quantify_specific_reaction(self, reaction_history, reaction_index):
        """
        Quantify a reaction from one simulation reaction history

        Args:
            reaction_history: array containing sequence of reactions fired during a simulation.
            reaction_index: integer of reaction index of interest

        :return: integer number of times reaction is fired
        """
        if reaction_index not in reaction_history:
            reaction_count = 0
        else:
            reaction_count = len(reaction_history[reaction_index])

        return reaction_count

    def quantify_rank_reactions(self, reaction_type=None, num_rxns=None):
        """
        Given reaction histories, identify the most commonly occurring reactions, on average.
        Can rank generally, or by reactions of a certain type.

        Args:
            reaction_profiles (list of dicts): reactions fired as a function of time
            reaction_type (string)
            num_rxns (int): the amount of reactions interested in collecting data on. If None, record for all.

        Returns:
            reaction_data: list of reactions and their avg, std of times fired. Sorted by the average times fired.
            [(rxn1, (avg, std)), (rxn2, (avg, std)) ... ]
        """
        allowed_rxn_types = [
            "One electron reduction",
            "One electron oxidation",
            "Intramolecular single bond breakage",
            "Intramolecular single bond formation",
            "Coordination bond breaking AM -> A+M",
            "Coordination bond forming A+M -> AM",
            "Molecular decomposition breaking one bond A -> B+C",
            "Molecular formation from one new bond A+B -> C",
            "Concerted",
        ]
        if reaction_type is not None:
            rxns_of_type = list()
            if reaction_type not in allowed_rxn_types:
                raise RuntimeError(
                    "This reaction type does not (yet) exist in our reaction networks."
                )

            for ind, rxn in enumerate(self.reaction_network.reactions):
                if rxn.reaction_type()["rxn_type_A"] == reaction_type:
                    rxns_of_type.append(2 * ind)
                elif rxn.reaction_type()["rxn_type_B"] == reaction_type:
                    rxns_of_type.append(2 * ind + 1)
        reaction_data = dict()  # keeping record of each iteration
        # Loop to count all reactions fired
        for n_sim in range(self.num_sims):
            rxns_fired = set(self.reaction_history[n_sim])
            if reaction_type is not None:
                relevant_rxns = [r for r in rxns_fired if r in rxns_of_type]
            else:
                relevant_rxns = rxns_fired

            for rxn_ind in relevant_rxns:
                if rxn_ind not in reaction_data:
                    reaction_data[rxn_ind] = list()
                reaction_data[rxn_ind].append(
                    np.sum(self.reaction_history[n_sim] == rxn_ind)
                )

        reaction_analysis = dict()
        for rxn_ind, counts in reaction_data.items():
            reaction_analysis[rxn_ind] = (
                np.mean(np.array(counts)),
                np.std(np.array(counts)),
            )

        # Sort reactions by the average amount fired
        sorted_reaction_analysis = sorted(
            [(i, c) for i, c in reaction_analysis.items()],
            key=lambda x: x[1][0],
            reverse=True,
        )
        if num_rxns is None:
            return sorted_reaction_analysis
        else:
            return sorted_reaction_analysis[:num_rxns]

    def frequency_analysis(self, rxn_inds, spec_inds, partitions=100):
        """
        Calculate the frequency of reaction and species formation as a function of time. Simulation data is
        discretized into time intervals, and probabilities in each set are obtained.

        Args:
            rxn_inds: list of indeces of reactions of interest
            spec_inds: list of molecule indexes of interest
            partitions: number of intervals in which to discretize time

        :return: dict of dicts containing the statistics of reaction fired, product formed at each time interval.
        {reaction_data: {rxn_ind1: [(t0, avg0, std0), (t1, avg1, std1), ...], rxn_ind2: [...], ... rxn_ind_n: [...]}
        {species_data: {spec1: [(t0, avg0, std0), (t1, avg1, std1), ...], spec2: [...], ... specn: [...]}}

        """
        reaction_frequency_data = dict()
        reaction_frequency_array = (
            dict()
        )  # Growing arrays of reaction frequencies as fxn of time
        species_frequency_data = dict()
        species_frequency_array = dict()
        new_species_counters = dict()
        for ind in rxn_inds:
            reaction_frequency_data[ind] = [0 for j in range(partitions)]

        for ind in spec_inds:
            species_frequency_data[ind] = [0 for j in range(partitions)]
            new_species_counters[ind] = 0

        for n_sim in range(self.num_sims):
            delta_t = np.sum(self.time_history[n_sim]) / partitions
            ind_0 = 0
            t = 0
            n = 0  # for tracking which time interval we are in
            species_counters = copy.deepcopy(
                new_species_counters
            )  # for counting species as they appear
            rxn_freq_data = copy.deepcopy(reaction_frequency_data)
            spec_freq_data = copy.deepcopy(species_frequency_data)
            for step_num, tau in enumerate(self.time_history[n_sim]):
                t += tau
                this_rxn_ind = int(self.reaction_history[n_sim][step_num])
                if this_rxn_ind % 2:  # reverse reaction
                    prods = self.reactants[math.floor(this_rxn_ind / 2), :]
                else:
                    prods = self.products[math.floor(this_rxn_ind / 2), :]

                for spec_ind in spec_inds:
                    if spec_ind in prods:
                        species_counters[spec_ind] += 1

                # When t reaches the next discretized time step, or end of the simulation
                if (t >= (n + 1) * delta_t) or (
                    step_num == len(self.reaction_history[n_sim]) - 1
                ):
                    n_to_fill = n
                    if t >= (n + 2) * delta_t:
                        n += math.floor(t / delta_t - n)
                    else:
                        n += 1
                    steps = step_num - ind_0 + 1
                    for spec_ind in spec_inds:
                        spec_freq_data[spec_ind][n_to_fill] = (
                            species_counters[spec_ind] / steps
                        )

                    for rxn_ind in rxn_inds:
                        rxn_freq = (
                            np.count_nonzero(
                                self.reaction_history[n_sim][ind_0 : step_num + 1]
                                == rxn_ind
                            )
                            / steps
                        )
                        # t_mdpt = (self.time_history[n_sim][step_num] + self.time_history[n_sim][ind_0]) / 2
                        rxn_freq_data[rxn_ind][n_to_fill] = rxn_freq

                    # Reset and update counters
                    species_counters = copy.deepcopy(new_species_counters)
                    ind_0 = step_num + 1

            for rxn_ind in rxn_inds:
                if n_sim == 0:
                    reaction_frequency_array[rxn_ind] = np.array(rxn_freq_data[rxn_ind])
                else:
                    reaction_frequency_array[rxn_ind] = np.vstack(
                        (reaction_frequency_array[rxn_ind], rxn_freq_data[rxn_ind])
                    )
            # print('reaction freq array', reaction_frequency_array)

            for spec_ind in spec_inds:
                if n_sim == 0:
                    species_frequency_array[spec_ind] = np.array(
                        spec_freq_data[spec_ind]
                    )
                else:
                    species_frequency_array[spec_ind] = np.vstack(
                        (species_frequency_array[spec_ind], spec_freq_data[spec_ind])
                    )
        # Statistical analysis
        statistical_rxn_data = dict()
        statistical_spec_data = dict()
        avg_delta_t = (
            np.mean(np.array([sum(self.time_history[i]) for i in range(self.num_sims)]))
            / partitions
        )
        time_list = [i * avg_delta_t + avg_delta_t / 2 for i in range(partitions)]
        # print('time_list: ', time_list)
        for rxn_ind in rxn_inds:
            if self.num_sims == 1:
                avgs = reaction_frequency_array[rxn_ind]
                stds = np.zeros(partitions)
            else:
                avgs = np.mean(reaction_frequency_array[rxn_ind], 0)
                stds = np.std(reaction_frequency_array[rxn_ind], 0)
            statistical_rxn_data[rxn_ind] = [
                (time_list[n], avgs[n], stds[n]) for n in range(partitions)
            ]

        for spec_ind in spec_inds:
            if self.num_sims == 1:
                spec_avgs = species_frequency_array[spec_ind]
                spec_stds = np.zeros(partitions)
            else:
                spec_avgs = np.mean(species_frequency_array[spec_ind], 0)
                spec_stds = np.std(species_frequency_array[spec_ind], 0)
            statistical_spec_data[spec_ind] = [
                (time_list[n], spec_avgs[n], spec_stds[n]) for n in range(partitions)
            ]

        return {
            "reaction_data": statistical_rxn_data,
            "species_data": statistical_spec_data,
        }

    def find_rxn_index(self, reaction, reverse):
        """
        Find the reaction index of a given reaction object

        Args:
            reaction: Reaction object
            reverse: bool to say whether reaction is reverse or forward
        :return: integer reaction index
        """
        for ind, rxn in enumerate(self.reaction_network.reactions):
            if rxn == reaction:
                if reverse is True:
                    rxn_ind = 2 * ind + 1
                else:
                    rxn_ind = 2 * ind
                break

        return rxn_ind
