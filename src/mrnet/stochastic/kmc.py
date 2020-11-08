# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import math
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A
from numba import jit
from numba import int64, float64

__author__ = "Ronald Kam, Evan Spotte-Smith, Xiaowei Xie"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

spec = [('products', int64[:, :]),
        ('reactants', int64[:, :]),
        ('initial_state', int64[:]),
        ('state', int64[:]),
        ('rate_constants', float64[:]),
        ('coord_array', float64[:]),
        ('species_rxn_mapping', int64[:, :]),
        ('volume', float64),
        ('num_species', int64),
        ('num_rxns', int64),
        ('rxn_ind', int64[:]),
        ('propensity_array', float64[:]),
        ('total_propensity', float64),
        ('state', int64[:]),
        ('times', float64[:]),
        ('reaction_history', int64[:])
        ]

"""Function-based KMC simulation for a reaction network, assuming spatial homogeneity. Simulation is performed without 
objects, such is required to improve performance with Numba. Algorithm is described by Gillespie (1976).
"""


def initialize_simulation(reaction_network, initial_cond, volume=10**-24):
    """Initial loop through reactions to create lists, mappings, and initial states needed for simulation without
    reaction network object.

    Args:
        reaction_network (ReactionNetwork): Fully generated reaction network
        initial_cond (dict): [mol_id: initial_conc [M] (float)]
        volume [m^3] (float)

    Returns:
        initial_state (list): Initial molecule amounts. Species indexing corresponds to reaction_network.entries_list
        initial_state_dict (dict): convert initial_cond to a dict of [initial_mol_ind: #molecules...]
        species_rxn_mapping (list of list): each species has a list of reaction (indexes) which they take part in
        molid_index_mapping (dict): mapping between species index and its Molecule entry id
                                    [molecule_index: molecule_entry_id]
        reactant_array (array) (n_rxns x 2): each row contains the reactant indexes of forward reaction
        products_array (array) (n_rxns x 2): each row contains the product indexes of forward reaction
        coord_array (array) (2*n_rxns x 1): coordination number for each forward and reverse reaction
                                            [c1_f, c1_r, c2_f, c2_r ...]
        rate_constants (array) (2*n_rxns x 1): rate constant of each for and rev reaction [k1_f, k1_r, k2_f, k2_r ...]
        propensities (array) (2*n_rxns x 1): reaction propensities, obtained by element-wise multiplication of
                                            coord_array and rate_constants
        molid_index_mapping (dict): [mol_id: mol_index (int) ... ]

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

    species_rxn_mapping_list = [[] for j in range(num_species)]
    reactant_array = -1 * np.ones((num_rxns, 2), dtype=int)
    product_array = -1 * np.ones((num_rxns, 2), dtype=int)
    coord_array = np.zeros(2 * num_rxns)
    rate_constants = np.zeros(2 * num_rxns)
    for id, reaction in enumerate(reaction_network.reactions):
        num_reactants_for = list()
        num_reactants_rev = list()
        rate_constants[2 * id] = reaction.rate_constant()["k_A"]
        rate_constants[2 * id + 1] = reaction.rate_constant()["k_B"]
        for idx, react in enumerate(reaction.reactants):
            # for each reactant, need to find the corresponding mol_id with the index
            mol_ind = molid_index_mapping[react.entry_id]
            reactant_array[id, idx] = mol_ind
            species_rxn_mapping_list[mol_ind].append(2 * id)
            num_reactants_for.append(initial_state[mol_ind])

        for idx, prod in enumerate(reaction.products):
            mol_ind = molid_index_mapping[prod.entry_id]
            product_array[id, idx] = mol_ind
            species_rxn_mapping_list[mol_ind].append(2*id + 1)
            num_reactants_rev.append(initial_state[mol_ind])

        if len(reaction.reactants) == 1:
            coord_array[2 * id] = num_reactants_for[0]
        elif (len(reaction.reactants) == 2) and (reaction.reactants[0] == reaction.reactants[1]):
            coord_array[2 * id] = num_reactants_for[0] * (num_reactants_for[0] - 1)
        elif (len(reaction.reactants) == 2) and (reaction.reactants[0] != reaction.reactants[1]):
            coord_array[2 * id] = num_reactants_for[0] * num_reactants_for[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        # For reverse reaction
        if len(reaction.products) == 1:
            coord_array[2 * id + 1] = num_reactants_rev[0]
        elif (len(reaction.products) == 2) and (reaction.products[0] == reaction.products[1]):
            coord_array[2 * id + 1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
        elif (len(reaction.products) == 2) and (reaction.products[0] != reaction.products[1]):
            coord_array[2 * id + 1] = num_reactants_rev[0] * num_reactants_rev[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
    rxn_mapping_lengths = [len(rxn_list) for rxn_list in species_rxn_mapping_list]
    max_mapping_length = max(rxn_mapping_lengths)
    species_rxn_mapping = -1 * np.ones((num_species, max_mapping_length), dtype=int)

    for index, rxn_list in enumerate(species_rxn_mapping_list):
        this_map_length = rxn_mapping_lengths[index]
        if this_map_length == max_mapping_length:
            species_rxn_mapping[index, :] = rxn_list
        else:
            species_rxn_mapping[index, : this_map_length - max_mapping_length] = rxn_list
    propensities = np.multiply(coord_array, rate_constants)
    return [initial_state, initial_state_dict, species_rxn_mapping, reactant_array, product_array,
            coord_array, rate_constants, propensities, molid_index_mapping]


@jit(nopython=True, parallel=True)
def kmc_simulate(time_steps, coord_array, rate_constants, propensity_array,
                 species_rxn_mapping, reactants, products, state):
    """ KMC Simulation of reaction network and specified initial conditions.

    Args:
         time_steps (int): Number of time steps/iterations desired to run.
         coord_array (array): Numpy array containing coordination numbers of forward and reverse reactions.
                                [h1f, h1r, h2f, h2r, ...]
         rate_constants (array): Numpy array containing rate constants of forward and reverse reactions.
         propensity_array (array): Numpy array containing propensities of for and rev reactions.
         species_rxn_mapping (2d array): Contains all the reaction indexes that each species takes part in
         reactants (2d array): Species IDs corresponding to the reactants of each forward reaction
         products (2d array): Species IDs corresponding to products of each forward reaction
         state (array): Array containing molecular amounts of each species in the reaction network

    Returns:
        A (2 x time_steps) Numpy array. First row contains the indeces of reactions that occurred.
        Second row are the time steps generated at each iterations.
    """
    total_propensity = np.sum(propensity_array)
    t = 0.0
    reaction_history = [0 for _ in range(time_steps)]
    times = [0.0 for _ in range(time_steps)]
    relevant_ind = np.where(propensity_array > 0)[0]
    for step_counter in range(time_steps):
        r1 = random.random()
        r2 = random.random()
        tau = -np.log(r1) / total_propensity
        random_propensity = r2 * total_propensity
        abrgd_reaction_choice_ind = np.where(np.cumsum(propensity_array[relevant_ind]) >= random_propensity)[0][0]
        reaction_choice_ind = relevant_ind[abrgd_reaction_choice_ind]
        converted_rxn_ind = math.floor(reaction_choice_ind / 2)
        if reaction_choice_ind % 2:
            reverse = True
        else:
            reverse = False
        state = update_state(reactants, products, state, converted_rxn_ind, reverse)
        # Log the reactions that need to be altered after reaction is performed, for the coordination array
        reactions_to_change: List[int64] = list()
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
            this_h = get_coordination(reactants, products, state, math.floor(rxn_ind/2), this_reverse)
            coord_array[rxn_ind] = this_h

        propensity_array = np.multiply(rate_constants, coord_array)
        relevant_ind = np.where(propensity_array > 0)[0]
        total_propensity = np.sum(propensity_array[relevant_ind])
        reaction_history[step_counter] = reaction_choice_ind
        times[step_counter] = tau
    return np.vstack((np.array(reaction_history), np.array(times)))


@jit(nopython=True)
def update_state(reactants, products, state, rxn_ind, reverse):
    """ Update the system based on the reaction chosen
            Args:
                reaction (Reaction)
                reverse (bool): If True, let the reverse reaction proceed.
                    Otherwise, let the forwards reaction proceed.

            Returns:
                None
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
                    raise ValueError("State invalid! Negative specie: {}!")
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
                    raise ValueError("State invalid! Negative specie: {}!")
        for product_id in products[rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                state[product_id] += 1
    return state


@jit(nopython=True)
def get_coordination(reactants, products, state, rxn_id, reverse):
    """ Calculate the coordination number for a particular reaction, based on the reaction type
    number of molecules for the reactants.

    Args:
        rxn_id (int): index of the reaction chosen in the array [R1f, R1r, R2f, R2r, ...]
        reverse (bool): If True, give the propensity for the reverse
            reaction. If False, give the propensity for the forwards
            reaction.

    Returns:
        propensity (float)
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
        raise RuntimeError("Only single and bimolecular reactions supported by this simulation")

    return h_prop


def plot_trajectory(reaction_network, molid_ind_mapping, initial_state_dict, products, reactants, reaction_history,
                    times, num_label, filename):
    """ Given lists of reaction history and time steps, iterate through and plot the data """
    cumulative_time = list(np.cumsum(np.array(times)))

    state = initial_state_dict

    state_to_plot = dict()
    for mol_ind in initial_state_dict:
        state_to_plot[mol_ind] = [(0.0, initial_state_dict[mol_ind])]
    total_iterations = len(reaction_history)

    for iter in range(total_iterations):
        this_rxn_ind = reaction_history[iter]
        converted_ind = math.floor(this_rxn_ind/2)
        t = cumulative_time[iter]

        if this_rxn_ind % 2:
            for r_ind in products[converted_ind, :]:
                if r_ind == -1:
                    continue
                else:
                    try:
                        state[r_ind] -= 1
                        if state[r_ind] < 0:
                            raise ValueError("State invalid: negative specie: {}".format(r_ind))
                        state_to_plot[r_ind].append((t, state[r_ind]))
                    except KeyError:
                        raise ValueError("Reactant specie {} given is not in state!".format(r_ind))
            for p_ind in reactants[converted_ind, :]:
                if p_ind == -1:
                    continue
                else:
                    if (p_ind in state) and (p_ind in state_to_plot):
                        state[p_ind] += 1
                        state_to_plot[p_ind].append((t, state[p_ind]))
                    else:
                        state[p_ind] = 1
                        state_to_plot[p_ind] = [(0.0, 0), (t, state[p_ind])]

        else:
            for r_ind in reactants[converted_ind, :]:
                if r_ind == -1:
                    continue
                else:
                    try:
                        state[r_ind] -= 1
                        if state[r_ind] < 0:
                            raise ValueError("State invalid: negative specie: {}".format(r_ind))
                        state_to_plot[r_ind].append((t, state[r_ind]))
                    except KeyError:
                        raise ValueError("Specie {} given is not in state!".format(r_ind))
            for p_ind in products[converted_ind, :]:
                if p_ind == -1:
                    continue
                elif (p_ind in state) and (p_ind in state_to_plot):
                    state[p_ind] += 1
                    state_to_plot[p_ind].append((t, state[p_ind]))
                else:
                    state[p_ind] = 1
                    state_to_plot[p_ind] = [(0.0, 0), (t, state[p_ind])]

    t_end = t
    # Sorting and plotting:
    fig, ax = plt.subplots()

    sorted_state = sorted([(k, v) for k, v in state.items()], key=lambda x: x[1], reverse=True)
    sorted_inds = [mol_tuple[0] for mol_tuple in sorted_state]

    sorted_ind_id_mapping = dict()
    iter_counter = 0
    for id, ind in molid_ind_mapping.items():

        if ind in sorted_inds[:num_label]:
            sorted_ind_id_mapping[ind] = id
            iter_counter += 1
        if iter_counter == num_label:
            break

    colors = plt.cm.get_cmap('hsv', num_label)
    this_id = 0

    for mol_ind in state_to_plot:
        ts = np.append(np.array([e[0] for e in state_to_plot[mol_ind]]), t_end)
        nums = np.append(np.array([e[1] for e in state_to_plot[mol_ind]]), state_to_plot[mol_ind][-1][1])
        if mol_ind in sorted_inds[:num_label]:
            mol_id = sorted_ind_id_mapping[mol_ind]
            for entry in reaction_network.entries_list:
                if mol_id == entry.entry_id:
                    this_composition = entry.molecule.composition.alphabetical_formula
                    this_charge = entry.molecule.charge
                    this_label = this_composition + " " + str(this_charge)
                    this_color = colors(this_id)
                    this_id += 1
                    break

            ax.plot(ts, nums, label=this_label, color=this_color)
        else:
            ax.plot(ts, nums)

    title = "KMC simulation, total time {}".format(cumulative_time[-1])
    ax.set(title=title,
           xlabel="Time (s)",
           ylabel="# Molecules")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
              ncol=2, fontsize="small")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def time_analysis(time_array):
    time_dict = dict()
    time_dict["t_avg"] = np.average(time_array)
    time_dict["t_std"] = np.std(time_array)
    time_dict["steps"] = len(time_array)
    time_dict["total_t"] = time_array[-1]  # minutes
    return time_dict
