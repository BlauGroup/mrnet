from typing import Tuple, Optional, Union, List, Dict, TextIO
import math
import numpy as np
import pickle
import os
import random
import sys

from mrnet.core.reactions import Reaction
from mrnet.network.reaction_network import ReactionNetwork
from mrnet.network.reaction_generation import ReactionGenerator
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.utils.visualization import (
    visualize_molecule_entry,
    visualize_molecule_count_histogram,
)

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender


def find_mol_entry_from_xyz_and_charge(mol_entries, xyz_file_path, charge):
    """
    given a file 'molecule.xyz', find the mol_entry corresponding to the
    molecule graph with given charge
    """
    target_mol_graph = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(xyz_file_path), OpenBabelNN()
    )

    # correction to the molecule graph
    target_mol_graph = metal_edge_extender(target_mol_graph)

    match = False
    index = -1
    while not match:
        index += 1
        mol_entry = mol_entries[index]
        species_mol_graph = mol_entry.mol_graph

        if mol_entry.charge == charge:
            match = target_mol_graph.isomorphic_to(species_mol_graph)

    if match:
        return mol_entry
    else:
        return None


# TODO: once there is a central place for these, import from there
boltzman_constant = 8.617e-5  # eV/K
planck_constant = 6.582e-16  # eV s
room_temp = 298.15  # K


class SerializedReactionNetwork:
    """
    An object designed to store data from a ReactionNetwork suitable for use with
    the C RNMC code.
    """

    def __init__(
        self,
        reaction_network: Union[ReactionNetwork, ReactionGenerator],
        logging: bool = False,
        temperature=room_temp,
        constant_barrier=None,
    ):

        if isinstance(reaction_network, ReactionGenerator):
            reactions = reaction_network
            entries_list = reaction_network.rn.entries_list

        else:
            reactions = reaction_network.reactions
            entries_list = reaction_network.entries_list

        self.logging = logging
        self.temperature = temperature
        self.constant_barrier = constant_barrier

        self.__extract_index_mappings(reactions)
        if logging:
            print("extracted index mappings")

        self.__extract_species_data(entries_list)
        if logging:
            print("extracted species data")

        if logging:
            print("set initial state")

        if logging:
            print("finished building serialization data")

    def internal_to_mrnet_index(self, internal_index):
        mol_entry = self.species_data[internal_index]
        return mol_entry.parameters["ind"]

    # if you are going to use this function heavily, probably best to
    # precompute a lookup dict here rather than looping through each time
    def mrnet_to_internal_index(self, mrnet_index):
        for internal_index, mol_entry in self.species_data.items():
            if mol_entry.parameters["ind"] == mrnet_index:
                return internal_index

    def mol_entry_to_internal_index(self, mol_entry):
        return self.species_to_index[mol_entry.entry_id]

    def __extract_index_mappings(self, reactions):
        """
        assign each species an index and construct
        forward and backward mappings between indicies and species.

        assign each reaction an index and construct
        a mapping from reaction indices to reaction data
        """
        species_to_index = {}
        index_to_reaction = []
        index = 0
        reaction_count = 0

        for reaction in reactions:
            reaction_count += 1
            entry_ids = {e.entry_id for e in reaction.reactants + reaction.products}
            for entry_id in entry_ids:
                species = entry_id
                if species not in species_to_index:
                    species_to_index[species] = index
                    index = index + 1

            reactant_indices = [
                species_to_index[reactant] for reactant in reaction.reactant_ids
            ]
            product_indices = [
                species_to_index[product] for product in reaction.product_ids
            ]

            forward_free_energy = reaction.free_energy_A
            backward_free_energy = reaction.free_energy_B

            index_to_reaction.append(
                {
                    "reactants": reactant_indices,
                    "products": product_indices,
                    "free_energy": forward_free_energy,
                }
            )
            index_to_reaction.append(
                {
                    "reactants": product_indices,
                    "products": reactant_indices,
                    "free_energy": backward_free_energy,
                }
            )

        for reaction in index_to_reaction:

            dG = reaction["free_energy"]
            kT = boltzman_constant * self.temperature
            max_rate = kT / planck_constant

            if self.constant_barrier is None:
                if dG < 0:
                    rate = max_rate
                else:
                    rate = max_rate * math.exp(-dG / kT)

            # if all rates are being set using a constant_barrier as in this formula,
            # then the constant barrier will not actually affect the simulation. It
            # becomes important when rates are being manually set.
            else:
                if dG < 0:
                    rate = max_rate * math.exp(-self.constant_barrier / kT)
                else:
                    rate = max_rate * math.exp(-(self.constant_barrier + dG) / kT)

            reaction["rate_constant"] = rate

        rev = {i: species for species, i in species_to_index.items()}
        self.number_of_reactions = 2 * reaction_count
        self.number_of_species = index
        self.species_to_index = species_to_index
        self.index_to_species = rev
        self.index_to_reaction = index_to_reaction

    def __extract_species_data(self, entries_list):
        """
        store MoleculeEntry data so it can be recalled later
        """
        species_data = {}
        for entry in entries_list:
            entry_id = entry.entry_id
            if entry_id in self.species_to_index:
                species_data[self.species_to_index[entry_id]] = entry

        self.species_data = species_data

    def serialize(
        self,
        folder: str,
        initial_state_data: List[Tuple[MoleculeEntry, int]],
        factor_zero: float = 1.0,
        factor_two: float = 1.0,
        factor_duplicate: float = 1.0,
    ):

        """
        write the reaction networks to files for ingestion by RNMC
        """

        # these variables are used like folder + number_of_species_postfix
        # postfix is to remind us that they are not total paths
        number_of_species_postfix = "/number_of_species"
        number_of_reactions_postfix = "/number_of_reactions"
        number_of_reactants_postfix = "/number_of_reactants"
        reactants_postfix = "/reactants"
        number_of_products_postfix = "/number_of_products"
        products_postfix = "/products"
        factor_zero_postfix = "/factor_zero"
        factor_two_postfix = "/factor_two"
        factor_duplicate_postfix = "/factor_duplicate"
        rates_postfix = "/rates"
        initial_state_postfix = "/initial_state"

        os.mkdir(folder)

        with open(folder + number_of_species_postfix, "w") as f:
            f.write(str(self.number_of_species) + "\n")

        with open(folder + number_of_reactions_postfix, "w") as f:
            f.write(str(self.number_of_reactions) + "\n")

        with open(folder + number_of_reactants_postfix, "w") as f:
            for reaction in self.index_to_reaction:
                f.write(str(len(reaction["reactants"])) + "\n")

        with open(folder + reactants_postfix, "w") as f:
            for reaction in self.index_to_reaction:
                for index in reaction["reactants"]:
                    f.write(str(index) + " ")
                f.write("\n")

        with open(folder + number_of_products_postfix, "w") as f:
            for reaction in self.index_to_reaction:
                f.write(str(len(reaction["products"])) + "\n")

        with open(folder + products_postfix, "w") as f:
            for reaction in self.index_to_reaction:
                for index in reaction["products"]:
                    f.write(str(index) + " ")
                f.write("\n")

        with open(folder + factor_two_postfix, "w") as f:
            f.write(("%e" % factor_two) + "\n")

        with open(folder + factor_zero_postfix, "w") as f:
            f.write(("%e" % factor_zero) + "\n")

        with open(folder + factor_duplicate_postfix, "w") as f:
            f.write(("%e" % factor_duplicate) + "\n")

        with open(folder + rates_postfix, "w") as f:
            for reaction in self.index_to_reaction:
                f.write(("%e" % reaction["rate_constant"]) + "\n")

        initial_state = np.zeros(self.number_of_species)
        for (mol_entry, count) in initial_state_data:
            index = self.mol_entry_to_internal_index(mol_entry)
            initial_state[index] = count

        with open(folder + initial_state_postfix, "w") as f:
            for i in range(self.number_of_species):
                f.write(str(int(initial_state[i])) + "\n")

        with open(folder + "/rnsd.pickle", "wb") as p:
            pickle.dump(self, p)

        print("finished serializing")


def serialize_simulation_parameters(
    folder: str,
    number_of_threads: int = 4,
    step_cutoff: Optional[int] = 200,
    time_cutoff: Optional[float] = None,
    number_of_simulations: int = 1000,
    base_seed: int = 1000,
):
    """
    write simulation paramaters to a file so that they can be ingested by RNMC

    """

    number_of_seeds_postfix = "/number_of_seeds"
    number_of_threads_postfix = "/number_of_threads"
    seeds_postfix = "/seeds"
    time_cutoff_postfix = "/time_cutoff"
    step_cutoff_postfix = "/step_cutoff"

    os.mkdir(folder)

    if step_cutoff is not None:
        with open(folder + step_cutoff_postfix, "w") as f:
            f.write(("%d" % step_cutoff) + "\n")
    elif time_cutoff is not None:
        with open(folder + time_cutoff_postfix, "w") as f:
            f.write(("%f" % time_cutoff) + "\n")
    else:
        raise ValueError("Either time_cutoff or step_cutoff must be set!")

    with open(folder + number_of_seeds_postfix, "w") as f:
        f.write(str(number_of_simulations) + "\n")

    with open(folder + number_of_threads_postfix, "w") as f:
        f.write(str(number_of_threads) + "\n")

    with open(folder + seeds_postfix, "w") as f:
        for seed in range(1000, 1000 + number_of_simulations * 2):
            f.write(str(seed) + "\n")


def run_simulator(network_folder, param_folder, path=None):
    if path is not None:
        os.system(path + " " + network_folder + " " + param_folder)
    else:
        os.system("RNMC " + network_folder + " " + param_folder)
