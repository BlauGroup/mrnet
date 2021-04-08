from typing import Tuple, Optional, Union, List
import math
import numpy as np
import pickle
import os
import sqlite3

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender

from mrnet.network.reaction_network import ReactionNetwork
from mrnet.network.reaction_generation import ReactionGenerator
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.utils.constants import ROOM_TEMP, PLANCK, KB


create_tables = """
    CREATE TABLE metadata (
            number_of_species   INTEGER NOT NULL,
            number_of_reactions INTEGER NOT NULL
    );

    CREATE TABLE reactions (
            reaction_id         INTEGER NOT NULL PRIMARY KEY,
            reaction_string     TEXT NOT NULL,
            number_of_reactants INTEGER NOT NULL,
            number_of_products  INTEGER NOT NULL,
            reactant_1          INTEGER NOT NULL,
            reactant_2          INTEGER NOT NULL,
            product_1           INTEGER NOT NULL,
            product_2           INTEGER NOT NULL,
            rate                REAL NOT NULL
    );

    CREATE UNIQUE INDEX reaction_string_idx ON reactions (reaction_string);
"""

insert_reaction = """
  INSERT INTO reactions (
          reaction_id,
          reaction_string,
          number_of_reactants,
          number_of_products,
          reactant_1,
          reactant_2,
          product_1,
          product_2,
          rate)
  VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);
"""

insert_metadata = """
  INSERT INTO metadata (
          number_of_species,
          number_of_reactions)
  VALUES (?1, ?2);
"""



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

def rate(dG, constant_barrier, temperature):
    kT = KB * temperature
    max_rate = kT / PLANCK

    if constant_barrier is None:
        if dG < 0:
            rate = max_rate
        else:
            rate = max_rate * math.exp(-dG / kT)

    # if all rates are being set using a constant_barrier as in this formula,
    # then the constant barrier will not actually affect the simulation. It
    # becomes important when rates are being manually set.
    else:
        if dG < 0:
            rate = max_rate * math.exp(-constant_barrier / kT)
        else:
            rate = max_rate * math.exp(-(constant_barrier + dG) / kT)

    return rate


class SerializedReactionNetwork:
    """
    An object designed to store data from a ReactionNetwork suitable for use with
    the C RNMC code.
    """

    def __init__(
        self,
        reaction_network: ReactionGenerator,
        logging: bool = False,
        temperature=ROOM_TEMP,
        constant_barrier=None,
    ):

        self.reactions = reaction_network
        self.entries_list = reaction_network.rn.entries_list

        self.logging = logging
        self.temperature = temperature
        self.constant_barrier = constant_barrier

    def serialize_initial_state(
        self,
        folder:str,
        initial_state_data: List[Tuple[MoleculeEntry, int]],
        factor_zero: float = 1.0,
        factor_two: float = 1.0,
        factor_duplicate: float = 1.0,
    ):

        factor_zero_postfix = "/factor_zero"
        factor_two_postfix = "/factor_two"
        factor_duplicate_postfix = "/factor_duplicate"
        initial_state_postfix = "/initial_state"


        with open(folder + factor_two_postfix, "w") as f:
            f.write(("%e" % factor_two) + "\n")

        with open(folder + factor_zero_postfix, "w") as f:
            f.write(("%e" % factor_zero) + "\n")

        with open(folder + factor_duplicate_postfix, "w") as f:
            f.write(("%e" % factor_duplicate) + "\n")

        initial_state = np.zeros(len(self.entries_list))
        for (mol_entry, count) in initial_state_data:
            index = mol_entry.parameters['ind']
            initial_state[index] = count

        with open(folder + initial_state_postfix, "w") as f:
            for i in range(len(initial_state)):
                f.write(str(int(initial_state[i])) + "\n")



    def serialize_network(
        self,
        folder: str,
    ):

        """
        write the reaction networks to files for ingestion by RNMC
        """

        # these variables are used like folder + number_of_species_postfix
        # postfix is to remind us that they are not total paths
        db_postfix = "/rn.sqlite"


        os.mkdir(folder)



        con = sqlite3.connect(folder + db_postfix)
        cur = con.cursor()
        cur.executescript(create_tables)
        con.commit()

        number_of_reactions = 0
        for reaction in self.reactions:

            reactant_strings = []
            product_strings = []
            try:
                reactant_1_index = reaction.reactants[0].parameters['ind']
                reactant_strings.append(str(reactant_1_index))
            except:
                reactant_1_index = -1

            try:
                reactant_2_index = reaction.reactants[1].parameters['ind']
                reactant_strings.append(str(reactant_2_index))
            except:
                reactant_2_index = -1

            try:
                product_1_index = reaction.products[0].parameters['ind']
                product_strings.append(str(product_1_index))
            except:
                product_1_index = -1

            try:
                product_2_index = reaction.products[1].parameters['ind']
                product_strings.append(str(product_2_index))

            except:
                product_2_index = -1

            forward_reaction_string = '+'.join(reactant_strings) + '->' + '+'.join(product_strings)
            reverse_reaction_string = '+'.join(product_strings) + '->' + '+'.join(reactant_strings)

            forward_free_energy = reaction.free_energy_A
            backward_free_energy = reaction.free_energy_B
            forward_rate = rate(
                forward_free_energy,
                self.constant_barrier,
                self.temperature)
            backward_rate = rate(
                backward_free_energy,
                self.constant_barrier,
                self.temperature)

            cur.execute(
                insert_reaction,
                ( number_of_reactions,
                  forward_reaction_string,
                  len(reactant_strings),
                  len(product_strings),
                  reactant_1_index,
                  reactant_2_index,
                  product_1_index,
                  product_2_index,
                  forward_rate))

            cur.execute(
                insert_reaction,
                ( number_of_reactions + 1,
                  reverse_reaction_string,
                  len(product_strings),
                  len(reactant_strings),
                  product_1_index,
                  product_2_index,
                  reactant_1_index,
                  reactant_2_index,
                  backward_rate))


            number_of_reactions += 2

            if number_of_reactions % 10000 == 0:
                con.commit()

        cur.execute(insert_metadata, (len(self.entries_list),number_of_reactions))
        con.commit()
        con.close()



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
