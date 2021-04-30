from typing import Tuple, Optional, Union, List, Dict
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


create_metadata_table = """
    CREATE TABLE metadata (
            number_of_species   INTEGER NOT NULL,
            number_of_reactions INTEGER NOT NULL,
            shard_size          INTEGER NOT NULL,
            number_of_shards    INTEGER NOT NULL
    );
"""


def create_reactions_table(n):
    return (
        "CREATE TABLE reactions_"
        + str(n)
        + """ (
                reaction_id         INTEGER NOT NULL PRIMARY KEY,
                reaction_string     TEXT NOT NULL,
                number_of_reactants INTEGER NOT NULL,
                number_of_products  INTEGER NOT NULL,
                reactant_1          INTEGER NOT NULL,
                reactant_2          INTEGER NOT NULL,
                product_1           INTEGER NOT NULL,
                product_2           INTEGER NOT NULL,
                rate                REAL NOT NULL,
                dG                  REAL NOT NULL
        );

CREATE INDEX reaction_"""
        + str(n)
        + "_string_idx ON reactions_"
        + str(n)
        + " (reaction_string);"
    )


def insert_reaction(n):
    return (
        "INSERT INTO reactions_"
        + str(n)
        + """ (
        reaction_id,
        reaction_string,
        number_of_reactants,
        number_of_products,
        reactant_1,
        reactant_2,
        product_1,
        product_2,
        rate,
        dG)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);
"""
    )


insert_metadata = """
  INSERT INTO metadata (
          number_of_species,
          number_of_reactions,
          shard_size,
          number_of_shards)
  VALUES (?1, ?2, ?3, ?4);
"""

def does_reaction_exist(n):
    return "SELECT COUNT(*) FROM reactions_" + str(n) + " WHERE reaction_string = ?"


class SerializeNetwork:
    """
    write the reaction network to a database for ingestion by RNMC.


    there are some non trivial aspects to the implementation. Firstly,
    we don't want to insert any duplicate reactions. This would be equivalent
    to doubling their rate constant.

    For this reason, we have the column reaction string in the reaction tables which
    has an index, so we can efficiently query the database to look for duplicate reactions.

    On the otherhand, we are going to be serializing reaction networks with hundreds of millions
    of reactions. Inserting rows into tables of this size which have indexes eventually grinds to a halt:
    https://stackoverflow.com/questions/15778716/sqlite-insert-speed-slows-as-number-of-records-increases-due-to-an-index

    to overcome this, we shard the reaction table into a bundle of smaller tables with shard_size rows.
    This creates a performance tradeoff. On the one hand, we want to keep the shard size small,
    so that insertions don't grind to a halt. But if the shard size is too small, we need to check for duplicates
    in more tables which also slows down the serialization process. Based on the above link, the optimal shard_size
    is probably somewhere between 1-3 million, but this is going to be hardware dependent.
    """

    def __init__(
        self,
        folder: str,
        reaction_generator: ReactionGenerator,
        shard_size: int = 1000000,
        commit_barrier: int = 10000,
        temperature=ROOM_TEMP,
        constant_barrier: float = 0.0,
    ):

        if shard_size < 0 or shard_size % 2 != 0:
            raise ValueError("shard_size must be positive and even")

        self.folder = folder
        self.reaction_generator = reaction_generator
        self.shard_size = shard_size
        self.commit_barrier = commit_barrier
        self.temperature = temperature
        self.constant_barrier = constant_barrier
        self.entries_list = self.reaction_generator.rn.entries_list
        self.db_postfix = "/rn.sqlite"
        self.current_shard = -1
        self.number_of_reactions = 0
        self.insert_statements: Dict[int, str] = {}
        self.does_exist_statements: Dict[int, str] = {}

        os.mkdir(self.folder)
        self.con = sqlite3.connect(self.folder + self.db_postfix)

        cur = self.con.cursor()
        cur.executescript(create_metadata_table)

        self.new_shard()
        self.serialize()

        self.number_of_shards = self.current_shard + 1
        cur.execute(
            insert_metadata,
            (
                len(self.entries_list),
                self.number_of_reactions,
                self.shard_size,
                self.number_of_shards,
            ),
        )

        self.con.commit()
        self.con.close()

    def new_shard(self):
        cur = self.con.cursor()
        self.current_shard += 1
        cur.executescript(create_reactions_table(self.current_shard))
        self.insert_statements[self.current_shard] = insert_reaction(self.current_shard)
        self.does_exist_statements[self.current_shard] = does_reaction_exist(
            self.current_shard
        )
        self.con.commit()


    def does_reaction_exist(self, reaction_string: str):
        cur = self.con.cursor()
        for i in range(self.current_shard + 1):
            cur.execute(self.does_exist_statements[i], (reaction_string,))
            count = cur.fetchone()
            if count[0] != 0:
                return True

        return False





    def insert_reaction(
        self,
        reaction_string,
        number_of_reactants,
        number_of_products,
        reactant_1,
        reactant_2,
        product_1,
        product_2,
        rate,
        free_energy,
    ):

        cur = self.con.cursor()

        shard = self.number_of_reactions // self.shard_size
        if shard > self.current_shard:
            self.new_shard()

        cur.execute(
            self.insert_statements[self.current_shard],
            (
                self.number_of_reactions,
                reaction_string,
                number_of_reactants,
                number_of_products,
                reactant_1,
                reactant_2,
                product_1,
                product_2,
                rate,
                free_energy,
            ),
        )

        self.number_of_reactions += 1

        if self.number_of_reactions % self.commit_barrier == 0:
            self.con.commit()

    def serialize(self):

        for (
            reactants,
            products,
            forward_free_energy,
            backward_free_energy,
        ) in self.reaction_generator:

            forward_reaction_string = "".join(
                [
                    "+".join([str(i) for i in reactants]),
                    "->",
                    "+".join([str(i) for i in products]),
                ]
            )
            if not self.does_reaction_exist(forward_reaction_string):
                reverse_reaction_string = "".join(
                    [
                        "+".join([str(i) for i in products]),
                        "->",
                        "+".join([str(i) for i in reactants]),
                    ]
                )

                try:
                    reactant_1_index = int(reactants[0])
                except IndexError:
                    reactant_1_index = -1

                try:
                    reactant_2_index = int(reactants[1])
                except IndexError:
                    reactant_2_index = -1

                try:
                    product_1_index = int(products[0])
                except IndexError:
                    product_1_index = -1

                try:
                    product_2_index = int(products[1])
                except IndexError:
                    product_2_index = -1

                forward_rate = rate(
                    forward_free_energy, self.temperature, self.constant_barrier
                )

                backward_rate = rate(
                    backward_free_energy, self.temperature, self.constant_barrier
                )

                self.insert_reaction(
                    forward_reaction_string,
                    len(reactants),
                    len(products),
                    reactant_1_index,
                    reactant_2_index,
                    product_1_index,
                    product_2_index,
                    forward_rate,
                    forward_free_energy,
                )

                self.insert_reaction(
                    reverse_reaction_string,
                    len(products),
                    len(reactants),
                    product_1_index,
                    product_2_index,
                    reactant_1_index,
                    reactant_2_index,
                    backward_rate,
                    backward_free_energy,
                )


def rate(dG, temperature, constant_barrier):
    kT = KB * temperature
    max_rate = kT / PLANCK

    # if all rates are being set using a constant_barrier > 0,
    # then the constant barrier will not actually affect the simulation. It
    # becomes important when rates are being manually set.
    if dG < 0:
        rate = max_rate * math.exp(-constant_barrier / kT)
    else:
        rate = max_rate * math.exp(-(constant_barrier + dG) / kT)

    return rate


def serialize_initial_state(
    folder: str,
    entries_list,
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

    initial_state = np.zeros(len(entries_list))
    for (mol_entry, count) in initial_state_data:
        index = mol_entry.parameters["ind"]
        initial_state[index] = count

    with open(folder + initial_state_postfix, "w") as f:
        for i in range(len(initial_state)):
            f.write(str(int(initial_state[i])) + "\n")


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


def clone_database(network_folder_1, network_folder_2):
    """
    serializing a network takes a long time, so instead of serializing twice
    we symlink the db from the first network folder into the second
    """
    db_postfix = "/rn.sqlite "
    os.system("mkdir " + network_folder_2)
    os.system("ln -s " + network_folder_1 + db_postfix + network_folder_2 + db_postfix)
