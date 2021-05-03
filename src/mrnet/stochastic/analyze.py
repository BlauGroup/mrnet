from typing import Tuple, List, Dict, TextIO
import pickle
import os
import copy
import sqlite3
from multiprocessing import Pool
import numpy as np
from functools import partial

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.utils.visualization import (
    visualize_molecule_entry,
    visualize_molecule_count_histogram,
    generate_latex_header,
    generate_latex_footer,
    latex_emit_molecule,
    latex_emit_reaction,
    visualize_molecules,
)

from mrnet.stochastic.serialize import rate


get_metadata = """
    SELECT * FROM metadata;
"""


def get_reaction(n: int):
    return (
        """
    SELECT reactant_1,
           reactant_2,
           product_1,
           product_2,
           dG
    FROM reactions_"""
        + str(n)
        + " WHERE reaction_id = ?;"
    )


def update_rate(shard: int):
    return (
        "UPDATE reactions_"
        + str(shard)
        + """
        SET rate = ?
        WHERE reaction_id = ?;
        """
    )


def does_reaction_exist(n):
    return (
        "SELECT reaction_id FROM reactions_" + str(n) + " WHERE reaction_string = ?1;"
    )


def get_reaction_string(n: int):
    return (
        """
    SELECT reaction_string
    FROM reactions_"""
        + str(n)
        + " WHERE reaction_id = ?1;"
    )


def find_duplicate_reactions(
    db_path: str,
    shard_size: int,
    number_of_shards: int,
    number_of_reactions: int,
    shard: int,
):
    repeats = []
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    get_reaction_string_sql = get_reaction_string(shard)

    does_reaction_exist_sql = []
    for i in range(number_of_shards):
        does_reaction_exist_sql.append(does_reaction_exist(i))

    base_index = shard * shard_size
    top_index = min(number_of_reactions, (shard + 1) * shard_size)
    for index in range(base_index, top_index):
        duplicate_indices = []
        reaction_string = list(cur.execute(get_reaction_string_sql, (index,)))[0][0]

        for sql in does_reaction_exist_sql:
            for row in cur.execute(sql, (reaction_string,)):
                duplicate_indices.append(row[0])

        if len(duplicate_indices) != 1:
            repeats.append(sorted(duplicate_indices))

    return repeats


class NetworkUpdater:
    """
    class to manage the state required for updating a sharded database.
    This could easily be a single function, but i anticipate that we will
    be adding more methods in the future.
    """

    def __init__(
        self, network_folder: str, number_of_threads=6  # used in duplicate checking
    ):

        self.network_folder = network_folder
        self.db_postfix = "/rn.sqlite"
        self.connection = sqlite3.connect(self.network_folder + self.db_postfix)
        self.number_of_threads = number_of_threads
        cur = self.connection.cursor()
        md = list(cur.execute(get_metadata))[0]
        self.number_of_species = md[0]
        self.number_of_reactions = md[1]
        self.shard_size = md[2]
        self.number_of_shards = md[3]
        self.update_rates_sql = {}
        self.get_reactions_sql = {}

        for i in range(self.number_of_shards):
            self.update_rates_sql[i] = update_rate(i)
            self.get_reactions_sql[i] = get_reaction(i)

    def update_rates(self, pairs: List[Tuple[int, float]]):
        cur = self.connection.cursor()
        for (index, r) in pairs:
            shard = index // self.shard_size
            cur.execute(self.update_rates_sql[shard], (r, index))

        self.connection.commit()

    def recompute_all_rates(
        self, temperature, constant_barrier, commit_frequency=10000
    ):
        cur = self.connection.cursor()

        for index in range(self.number_of_reactions):
            shard = index // self.shard_size
            res = list(cur.execute(self.get_reactions_sql[shard], (int(index),)))[0]
            dG = res[4]
            new_rate = rate(dG, temperature, constant_barrier)
            cur.execute(self.update_rates_sql[shard], (new_rate, index))

            if index % commit_frequency == 0:
                self.connection.commit()

        self.connection.commit()

    def find_duplicates(self):
        f = partial(
            find_duplicate_reactions,
            self.network_folder + self.db_postfix,
            self.shard_size,
            self.number_of_shards,
            self.number_of_reactions,
        )

        with Pool(self.number_of_threads) as p:
            repeats_unordered = p.map(f, range(self.number_of_shards))

        repeated = set()

        for xs in repeats_unordered:
            for x in xs:
                repeated.add(tuple(sorted(x)))

        return repeated

    def set_duplicate_reaction_rates_to_zero(self):
        repeats = self.find_duplicates()
        update_list = []
        for xs in repeats:
            head = True
            for x in xs:
                if head:
                    head = False
                else:
                    update_list.append((x, 0.0))

        self.update_rates(update_list)


def collect_duplicate_pathways(pathways: List[List[int]]) -> Dict[frozenset, dict]:
    pathway_dict: Dict[frozenset, dict] = {}
    for pathway in pathways:
        key = frozenset(pathway)
        if key in pathway_dict:
            pathway_dict[key]["frequency"] += 1
        else:
            pathway_dict[key] = {"pathway": pathway, "frequency": 1}
    return pathway_dict


def update_state(state, reaction):
    for species_index in reaction["reactants"]:
        state[species_index] -= 1

    for species_index in reaction["products"]:
        state[species_index] += 1


class SimulationAnalyzer:
    """
    A class to analyze the resutls of a set of MC runs
    """

    def __init__(self, network_folder: str, mol_list: List[MoleculeEntry]):

        initial_state_postfix = "/initial_state"
        simulation_histories_postfix = "/simulation_histories"
        database_postfix = "/rn.sqlite"
        reports_postfix = "/reports"

        self.connection = sqlite3.connect(network_folder + database_postfix)
        cur = self.connection.cursor()
        md = list(cur.execute(get_metadata))[0]
        self.number_of_species = md[0]
        self.number_of_reactions = md[1]
        self.shard_size = md[2]
        self.number_of_shards = md[3]
        self.get_reactions_sql = {}

        for i in range(self.number_of_shards):
            self.get_reactions_sql[i] = get_reaction(i)

        self.network_folder = network_folder
        self.histories_folder = network_folder + simulation_histories_postfix
        self.reports_folder = network_folder + reports_postfix

        try:
            os.mkdir(self.reports_folder)
        except FileExistsError:
            pass

        with open(network_folder + initial_state_postfix, "r") as f:
            initial_state_list = [int(c) for c in f.readlines()]
            self.initial_state = np.array(initial_state_list, dtype=int)

        self.mol_entries = {}

        for entry in mol_list:
            self.mol_entries[entry.parameters["ind"]] = entry

        self.reaction_data: Dict[int, dict] = {}

        self.reaction_pathways_dict: Dict[int, Dict[frozenset, dict]] = dict()
        self.reaction_histories = list()
        self.time_histories = list()
        self.observed_reactions = {}

        histories_contents = sorted(os.listdir(self.histories_folder))
        reaction_histories_contents = [
            x for x in histories_contents if x.startswith("reactions")
        ]
        time_histories_contents = [
            x for x in histories_contents if x.startswith("times")
        ]

        reaction_seeds = [x.split("_")[1] for x in reaction_histories_contents]
        time_seeds = [x.split("_")[1] for x in reaction_histories_contents]

        if reaction_seeds != time_seeds:
            raise ValueError("Reactions and times not from same set of initial seeds!")

        for filename in reaction_histories_contents:
            reaction_history = list()
            with open(self.histories_folder + "/" + filename) as f:
                for line in f:
                    reaction_history.append(int(line.strip()))

            self.reaction_histories.append(np.array(reaction_history))

        for filename in time_histories_contents:
            time_history = list()
            with open(self.histories_folder + "/" + filename) as f:
                for line in f:
                    time_history.append(float(line.strip()))

            self.time_histories.append(np.array(time_history))

        self.number_simulations = len(self.reaction_histories)
        visualize_molecules(
            self.reports_folder + "/molecule_diagrams", self.mol_entries
        )

    def index_to_reaction(self, reaction_index):
        shard = reaction_index // self.shard_size
        if reaction_index in self.reaction_data:
            return self.reaction_data[reaction_index]
        else:
            print("fetching data for reaction", reaction_index)
            cur = self.connection.cursor()
            # reaction_index is type numpy.int64 which sqlite doesn't like.
            res = list(
                cur.execute(self.get_reactions_sql[shard], (int(reaction_index),))
            )[0]
            reaction = {}
            reaction["reactants"] = [i for i in res[0:2] if i >= 0]
            reaction["products"] = [i for i in res[2:4] if i >= 0]
            reaction["dG"] = res[4]
            self.reaction_data[reaction_index] = reaction
            return reaction

    def extract_species_consumption_info(
        self, target_species_index: int
    ) -> Tuple[Dict[int, int], Dict[int, int], List[int]]:
        """
        given a target molecule, return all the ways the molecule was
        created, all the ways the molecule was consumed and the ending
        frequencies of the molecule for each simulation.
        """
        # if a reaction has the target species twice as a reactant or product
        # it will be counted twice
        producing_reactions = {}
        consuming_reactions = {}
        final_counts = []
        for reaction_history in self.reaction_histories:
            running_count = self.initial_state[target_species_index]

            for reaction_index in reaction_history:
                reaction = self.index_to_reaction(reaction_index)

                for reactant_index in reaction["reactants"]:
                    if target_species_index == reactant_index:
                        running_count -= 1
                        if reaction_index not in consuming_reactions:
                            consuming_reactions[reaction_index] = 1
                        else:
                            consuming_reactions[reaction_index] += 1

                for product_index in reaction["products"]:
                    if target_species_index == product_index:
                        running_count += 1
                        if reaction_index not in producing_reactions:
                            producing_reactions[reaction_index] = 1
                        else:
                            producing_reactions[reaction_index] += 1

            final_counts.append(running_count)

        return producing_reactions, consuming_reactions, final_counts

    def extract_reaction_pathways(self, target_species_index: int):
        """
        given a reaction history and a target molecule, find the
        first reaction which produced the target molecule (if any).
        Apply that reaction to the initial state to produce a partial
        state array. Missing reactants have negative values in the
        partial state array. Now loop through the reaction history
        to resolve the missing reactants.
        """

        print("extracting pathways to", target_species_index)
        reaction_pathway_list = []
        for reaction_history_num, reaction_history in enumerate(
            self.reaction_histories
        ):
            # current approach is a hack. Sometimes it can fall into an inifite loop
            # if pathway gets too long, we assume that this has happened.
            infinite_loop = False
            print("scanning history", reaction_history_num, "for pathway")

            # -1 if target wasn't produced
            # index of reaction if target was produced
            reaction_producing_target_index = -1
            for reaction_index in reaction_history:
                reaction = self.index_to_reaction(reaction_index)
                if target_species_index in reaction["products"]:
                    reaction_producing_target_index = reaction_index
                    break

            if reaction_producing_target_index == -1:
                continue
            else:
                pathway = [reaction_producing_target_index]
                partial_state = np.copy(self.initial_state)
                final_reaction = self.index_to_reaction(pathway[0])
                update_state(partial_state, final_reaction)

                negative_species = list(np.where(partial_state < 0)[0])

                while len(negative_species) != 0:
                    if len(pathway) > 1000:
                        infinite_loop = True
                        break
                    for species_index in negative_species:
                        for reaction_index in reaction_history:
                            reaction = self.index_to_reaction(reaction_index)
                            if species_index in reaction["products"]:
                                update_state(partial_state, reaction)
                                pathway.insert(0, reaction_index)
                                break

                    negative_species = list(np.where(partial_state < 0)[0])

                if not infinite_loop:
                    reaction_pathway_list.append(pathway)

        reaction_pathway_dict = collect_duplicate_pathways(reaction_pathway_list)
        self.reaction_pathways_dict[target_species_index] = reaction_pathway_dict

    def generate_consumption_report(self, mol_entry: MoleculeEntry):
        target_species_index = mol_entry.parameters["ind"]

        (
            producing_reactions,
            consuming_reactions,
            final_counts,
        ) = self.extract_species_consumption_info(target_species_index)

        histogram_file = (
            self.reports_folder
            + "/final_count_histogram_"
            + str(target_species_index)
            + ".pdf"
        )

        visualize_molecule_count_histogram(final_counts, histogram_file)

        with open(
            self.reports_folder
            + "/consumption_report_"
            + str(target_species_index)
            + ".tex",
            "w",
        ) as f:

            generate_latex_header(f)

            f.write("consumption report for")
            latex_emit_molecule(f, target_species_index)
            f.write("\n\n")

            f.write("molecule frequency at end of simulations")
            f.write(
                "\\raisebox{-.5\\height}{"
                + "\\includegraphics[scale=0.5]{"
                + "./final_count_histogram_"
                + str(target_species_index)
                + ".pdf"
                + "}}\n\n"
            )

            f.write("producing reactions:\n\n\n")

            for reaction_index, frequency in sorted(
                producing_reactions.items(), key=lambda item: -item[1]
            ):

                f.write(str(frequency) + " occurrences:\n")

                self.latex_emit_reaction(f, reaction_index)

            f.write("consuming reactions:\n\n\n")

            for reaction_index, frequency in sorted(
                consuming_reactions.items(), key=lambda item: -item[1]
            ):

                f.write(str(frequency) + " occurrences:\n")
                self.latex_emit_reaction(f, reaction_index)

            generate_latex_footer(f)

    def generate_pathway_report(self, mol_entry: MoleculeEntry, min_frequency: int):
        target_species_index = mol_entry.parameters["ind"]

        if target_species_index not in self.reaction_pathways_dict:
            self.extract_reaction_pathways(target_species_index)

        with open(
            self.reports_folder
            + "/pathway_report_"
            + str(target_species_index)
            + ".tex",
            "w",
        ) as f:

            pathways = self.reaction_pathways_dict[target_species_index]

            generate_latex_header(f)

            f.write("pathway report for\n\n")
            latex_emit_molecule(f, target_species_index)
            self.latex_emit_initial_state(f)

            f.write("\\newpage\n\n\n")

            for _, unique_pathway in sorted(
                pathways.items(), key=lambda item: -item[1]["frequency"]
            ):

                frequency = unique_pathway["frequency"]
                if frequency > min_frequency:
                    f.write(str(frequency) + " occurrences:\n")

                    for reaction_index in unique_pathway["pathway"]:
                        self.latex_emit_reaction(f, reaction_index)

                    f.write("\\newpage\n")
                else:
                    break

            generate_latex_footer(f)

    def latex_emit_initial_state(self, f: TextIO):
        f.write("\n\n initial state:\n\n\n")
        for species_index in range(self.number_of_species):
            num = self.initial_state[species_index]
            if num > 0:
                f.write(str(num) + " molecules of ")
                latex_emit_molecule(f, species_index)
                f.write("\n\n")

    def latex_emit_reaction(self, f: TextIO, reaction_index: int):
        reaction = self.index_to_reaction(reaction_index)
        latex_emit_reaction(f, reaction, reaction_index)

    def generate_simulation_history_report(self, history_num):
        with open(
            self.reports_folder
            + "/simulation_history_report_"
            + str(history_num)
            + ".tex",
            "w",
        ) as f:

            generate_latex_header(f)

            f.write("simulation " + str(history_num))
            f.write("\n\n\n")
            for reaction_index in self.reaction_histories[history_num]:
                f.write("\n\n\n")
                self.latex_emit_reaction(f, reaction_index)

            generate_latex_footer(f)

    def generate_list_of_all_reactions_report(self):
        with open(
            self.reports_folder + "/list_of_all_reactions.tex",
            "w",
        ) as f:

            generate_latex_header(f)

            for reaction_index in range(self.number_of_reactions):
                f.write("\n\n\n")
                self.latex_emit_reaction(f, reaction_index)

            generate_latex_footer(f)

    def generate_list_of_all_species_report(self):
        with open(
            self.reports_folder + "/list_of_all_species.tex",
            "w",
        ) as f:

            generate_latex_header(f)

            for species_index in range(self.number_of_species):
                f.write("\n\n\n")
                latex_emit_molecule(f, species_index)

            generate_latex_footer(f)

    def compute_reaction_tally(self):
        if len(self.observed_reactions) == 0:
            for history in self.reaction_histories:
                for reaction_index in history:
                    if reaction_index in self.observed_reactions:
                        self.observed_reactions[reaction_index] += 1
                    else:
                        self.observed_reactions[reaction_index] = 1


    def frequently_occouring_reactions(self, number: int):
        """
        return a list of the number most frequently occouring reactions
        """
        self.compute_reaction_tally()
        return list(
            map(lambda pair: pair[0],
                sorted(self.observed_reactions.items(), key=lambda pair: -pair[1])[0:number]))


    def generate_reaction_tally_report(self, cutoff: int):
        self.compute_reaction_tally()

        with open(self.reports_folder + "/reaction_tally_report.tex", "w") as f:

            generate_latex_header(f)

            f.write("reaction tally report")
            f.write("\n\n\n")
            for (reaction_index, number) in sorted(
                self.observed_reactions.items(), key=lambda pair: -pair[1]
            ):
                if number > cutoff:
                    f.write(str(number) + " occourances of:")
                    self.latex_emit_reaction(f, reaction_index)

            generate_latex_footer(f)

    def generate_time_dep_profiles(self, frequency: int = 1):
        """
        Generate plottable time-dependent profiles of species and rxns from raw KMC output, obtain final states.

        :param frequency (int): The system state will be sampled after every n
            reactions, where n is the frequency. Default is 1, meaning that each
            step will be sampled.

        :return dict containing species profiles, reaction profiles, and final states from each simulation.
                {species_profiles: [ {mol_ind1: [n(t0), n(t1)...], mol_ind2: [...],  ... }, {...}, ... ]
                reaction_profiles: [ {rxn_ind1: [n(t0), n(t1)...], rxn_ind2: [...],  ...}, {...}, ...]
                final_states: [ {mol_ind1: n1, mol_ind2: ..., ...}, {...}, ...],
                snapshot_times: [[t0, t1, ...], [...], ...]}

        """
        species_profiles = list()
        reaction_profiles = list()
        snapshot_times = list()
        final_states = list()

        for n_sim in range(self.number_simulations):
            sim_time_history = self.time_histories[n_sim]
            sim_rxn_history = self.reaction_histories[n_sim]
            state = copy.deepcopy(self.initial_state)
            rxn_counts = dict()
            snaps = [0.0]
            sim_species_profile = dict()
            sim_rxn_profile = dict()
            for ii, mol_ind in enumerate(state):
                sim_species_profile[ii] = [self.initial_state[ii]]
            for index in range(self.number_of_reactions):
                sim_rxn_profile[index] = [0]
                rxn_counts[index] = 0
            total_iterations = len(sim_rxn_history)

            for iter in range(total_iterations):
                rxn_ind = sim_rxn_history[iter]
                t = sim_time_history[iter]
                rxn_counts[rxn_ind] += 1

                update_state(state, self.index_to_reaction(rxn_ind))
                for i, v in enumerate(state):
                    if v < 0:
                        raise ValueError(
                            "State invalid: simulation {}, negative specie {}, time {}, step {}, reaction {}".format(
                                n_sim, i, t, iter, rxn_ind
                            )
                        )

                if iter + 1 % frequency == 0:
                    snaps.append(t)
                    for i, v in enumerate(state):
                        sim_species_profile[i].append(v)
                    for rxn, count in rxn_counts.items():
                        sim_rxn_profile[rxn].append(count)

            # Always add the final state
            if sim_time_history[-1] not in snaps:
                snaps.append(sim_time_history[-1])
                for i, v in enumerate(state):
                    sim_species_profile[i].append(v)
                for rxn, count in rxn_counts.items():
                    sim_rxn_profile[rxn].append(count)

            species_profiles.append(sim_species_profile)
            reaction_profiles.append(sim_rxn_profile)
            final_states.append(state)
            snapshot_times.append(snaps)

        return {
            "species_profiles": species_profiles,
            "reaction_profiles": reaction_profiles,
            "final_states": final_states,
            "snapshot_times": snapshot_times,
        }

    def final_state_analysis(self, final_states):
        """
        Gather statistical analysis of the final states of simulation.

        Args:
            final_states: list of dicts of final states, as generated in generate_time_dep_profiles()

        :return: list of tuples containing statistical data for each species, sorted from highest to low avg occurrence
        """

        # For each molecule, compile an array of its final amounts
        state_arrays = dict()
        for iter, final_state in enumerate(final_states):
            for index, amt in enumerate(final_state):
                # Store the amount, and convert key from mol_ind to entry_id
                if index not in state_arrays:
                    state_arrays[index] = np.zeros(self.number_simulations)
                state_arrays[index][iter] = amt
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

    def rank_reaction_counts(self):
        """
        Given reaction histories, identify the most commonly occurring reactions, on average.
        Can rank generally, or by reactions of a certain type.

        Args:

        Returns:
            reaction_data: list of reactions and their avg, std of times fired. Sorted by the average times fired.
            [(rxn1, (avg, std)), (rxn2, (avg, std)) ... ]
        """

        reaction_data = dict()  # keeping record of each iteration
        # Loop to count all reactions fired
        for n_sim in range(self.number_simulations):
            rxns_fired = set(self.reaction_histories[n_sim])

            for rxn_ind in rxns_fired:
                if rxn_ind not in reaction_data:
                    reaction_data[rxn_ind] = list()
                reaction_data[rxn_ind].append(
                    np.sum(self.reaction_histories[n_sim] == rxn_ind)
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

        return sorted_reaction_analysis
