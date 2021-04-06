from typing import Tuple, List, Dict, TextIO
import pickle
import os
import copy

import numpy as np

from mrnet.stochastic.serialize import SerializedReactionNetwork
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.utils.visualization import (
    visualize_molecule_entry,
    visualize_molecule_count_histogram,
)


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

    def __init__(
        self, rnsd: SerializedReactionNetwork, initial_state, network_folder: str
    ):
        """
        Params:
            rnsd (SerializedReactionNetwork):
            network_folder (Path):
        """

        self.network_folder = network_folder
        self.histories_folder = network_folder + "/simulation_histories"
        self.rnsd = rnsd
        self.initial_state = initial_state
        self.reaction_pathways_dict: Dict[int, Dict[frozenset, dict]] = dict()
        self.reaction_histories = list()
        self.time_histories = list()

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
        self.visualize_molecules()

    def visualize_molecules(self):
        folder = self.network_folder + "/molecule_diagrams"
        if os.path.isdir(folder):
            return

        os.mkdir(folder)
        for index in range(self.rnsd.number_of_species):
            molecule_entry = self.rnsd.species_data[index]
            visualize_molecule_entry(molecule_entry, folder + "/" + str(index) + ".pdf")

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
                reaction = self.rnsd.index_to_reaction[reaction_index]

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
        reaction_pathway_list = []
        for reaction_history in self.reaction_histories:

            # -1 if target wasn't produced
            # index of reaction if target was produced
            reaction_producing_target_index = -1
            for reaction_index in reaction_history:
                reaction = self.rnsd.index_to_reaction[reaction_index]
                if target_species_index in reaction["products"]:
                    reaction_producing_target_index = reaction_index
                    break

            if reaction_producing_target_index == -1:
                continue
            else:
                pathway = [reaction_producing_target_index]
                partial_state = np.copy(self.initial_state)
                final_reaction = self.rnsd.index_to_reaction[pathway[0]]
                update_state(partial_state, final_reaction)

                negative_species = list(np.where(partial_state < 0)[0])

                while len(negative_species) != 0:
                    for species_index in negative_species:
                        for reaction_index in reaction_history:
                            reaction = self.rnsd.index_to_reaction[reaction_index]
                            if species_index in reaction["products"]:
                                update_state(partial_state, reaction)
                                pathway.insert(0, reaction_index)
                                break

                    negative_species = list(np.where(partial_state < 0)[0])

                reaction_pathway_list.append(pathway)

        reaction_pathway_dict = collect_duplicate_pathways(reaction_pathway_list)
        self.reaction_pathways_dict[target_species_index] = reaction_pathway_dict

    def generate_consumption_report(self, mol_entry: MoleculeEntry):
        target_species_index = self.rnsd.mol_entry_to_internal_index(mol_entry)
        folder = (
            self.network_folder + "/consumption_report_" + str(target_species_index)
        )
        os.mkdir(folder)

        (
            producing_reactions,
            consuming_reactions,
            final_counts,
        ) = self.extract_species_consumption_info(target_species_index)

        visualize_molecule_count_histogram(
            final_counts, folder + "/final_count_histogram.pdf"
        )

        with open(folder + "/consumption_report.tex", "w") as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage[margin=1cm]{geometry}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\pagenumbering{gobble}\n")
            f.write("\\begin{document}\n")

            f.write("consumption report for")
            f.write(
                "\\raisebox{-.5\\height}{"
                + "\\includegraphics[scale=0.2]{../molecule_diagrams/"
                + str(target_species_index)
                + ".pdf}}\n\n"
            )

            f.write("molecule frequency at end of simulations")
            f.write(
                "\\raisebox{-.5\\height}{"
                + "\\includegraphics[scale=0.5]{./final_count_histogram.pdf"
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

            f.write("\\end{document}")

    def generate_pathway_report(self, mol_entry: MoleculeEntry, min_frequency: int):
        target_species_index = self.rnsd.mol_entry_to_internal_index(mol_entry)
        folder = self.network_folder + "/pathway_report_" + str(target_species_index)
        os.mkdir(folder)

        with open(folder + "/pathway_report.tex", "w") as f:
            if target_species_index not in self.reaction_pathways_dict:
                self.extract_reaction_pathways(target_species_index)

            pathways = self.reaction_pathways_dict[target_species_index]

            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage[margin=1cm]{geometry}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\pagenumbering{gobble}\n")
            f.write("\\begin{document}\n")

            f.write("pathway report for")
            f.write(
                "\\raisebox{-.5\\height}{"
                + "\\includegraphics[scale=0.2]{../molecule_diagrams/"
                + str(target_species_index)
                + ".pdf}}\n\n"
            )
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

            f.write("\\end{document}")

    def latex_emit_initial_state(self, f: TextIO):
        f.write("initial state:\n\n\n")
        for species_index in range(self.rnsd.number_of_species):
            num = self.initial_state[species_index]
            if num > 0:
                f.write(str(num) + " of ")
                f.write(
                    "\\raisebox{-.5\\height}{"
                    + "\\includegraphics[scale=0.2]{../molecule_diagrams/"
                    + str(species_index)
                    + ".pdf}}\n\n\n"
                )

    def latex_emit_reaction(self, f: TextIO, reaction_index: int):
        f.write("$$\n")
        reaction = self.rnsd.index_to_reaction[reaction_index]
        first = True
        for reactant_index in reaction["reactants"]:
            if first:
                first = False
            else:
                f.write("+\n")

            f.write(
                "\\raisebox{-.5\\height}{"
                + "\\includegraphics[scale=0.2]{../molecule_diagrams/"
                + str(reactant_index)
                + ".pdf}}\n"
            )

            # these are mrnet indices, which differ from the internal
            # MC indices
            mrnet_index = self.rnsd.internal_to_mrnet_index(reactant_index)
            f.write(str(mrnet_index) + "\n")

        f.write("\\xrightarrow{" + ("%.2f" % reaction["free_energy"]) + "}\n")

        first = True
        for product_index in reaction["products"]:
            if first:
                first = False
            else:
                f.write("+\n")

            f.write(
                "\\raisebox{-.5\\height}{"
                + "\\includegraphics[scale=0.2]{../molecule_diagrams/"
                + str(product_index)
                + ".pdf}}\n"
            )

            # these are mrnet indices, which differ from the internal
            # MC indices
            mrnet_index = self.rnsd.internal_to_mrnet_index(product_index)
            f.write(str(mrnet_index) + "\n")

        f.write("$$")
        f.write("\n\n\n")

    def generate_reaction_tally_report(self):
        observed_reactions = {}
        for history in self.reaction_histories:
            for reaction_index in history:
                if reaction_index in observed_reactions:
                    observed_reactions[reaction_index] += 1
                else:
                    observed_reactions[reaction_index] = 1

        folder = self.network_folder + "/reaction_tally_report"
        os.mkdir(folder)
        with open(folder + "/reaction_tally_report.tex", "w") as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage[margin=1cm]{geometry}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\pagenumbering{gobble}\n")
            f.write("\\begin{document}\n")

            f.write("reaction tally report")
            f.write("\n\n\n")
            for (reaction_index, number) in sorted(
                observed_reactions.items(), key=lambda pair: -pair[1]
            ):
                f.write(str(number) + " occourances of:")
                self.latex_emit_reaction(f, reaction_index)
            f.write("\\end{document}")

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
            for mol_ind in state:
                sim_species_profile[mol_ind] = [self.initial_state[mol_ind]]
            for index in self.rnsd.index_to_species:
                if index not in sim_species_profile:
                    sim_species_profile[index] = [0]
                if index not in state:
                    state[index] = 0
            for index in range(len(self.rnsd.index_to_reaction)):
                sim_rxn_profile[index] = [0]
                rxn_counts[index] = 0
            total_iterations = len(sim_rxn_history)

            for iter in range(total_iterations):
                rxn_ind = sim_rxn_history[iter]
                t = sim_time_history[iter]
                rxn_counts[rxn_ind] += 1

                update_state(state, rxn_ind)
                for i, v in state.items():
                    if v < 0:
                        raise ValueError("State invalid: negative specie {}".format(i))

                if iter + 1 % frequency == 0:
                    snaps.append(t)
                    for i, v in state.items():
                        sim_species_profile[i].append(v)
                    for rxn, count in rxn_counts.items():
                        sim_rxn_profile[rxn].append(count)

            # Always add the final state
            if sim_time_history[-1] not in snaps:
                snaps.append(sim_time_history[-1])
                for i, v in state.items():
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
            "snapshot_times": snapshot_times
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
            for index, amt in final_state.items():
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


def load_analysis(network_folder: str) -> SimulationAnalyzer:
    """
    as part of serialization, the SerializedReactionNetwork is stored as a
    pickle in the network folder. This allows for analysis to be picked up in a
    new python session.
    """
    with open(network_folder + "/rnsd.pickle", "rb") as f:
        rnsd = pickle.load(f)

    with open(network_folder + "/initial_state", "r") as s:
        initial_state = np.array([int(x) for x in s.readlines()], dtype=int)

    sa = SimulationAnalyzer(rnsd, initial_state, network_folder)

    return sa