from typing import Tuple, Optional, Union, List, Dict, TextIO
import numpy as np
import pickle
import os
import copy
import math

import matplotlib.pyplot as plt

from mrnet.core.reactions import Reaction
from mrnet.network.reaction_network import ReactionNetwork
from mrnet.network.reaction_generation import ReactionGenerator
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
                {species_profiles: [ {mol_ind1: [(t0, n(t0)), (t1, n(t1)...], mol_ind2: [...] ,  ... }, {...}, ... ]
                reaction_profiles: [ {rxn_ind1: [t0, t1, ...], rxn_ind2: ..., ...}, {...}, ...]
                final_states: [ {mol_ind1: n1, mol_ind2: ..., ...}, {...}, ...] }

        """
        species_profiles = list()
        reaction_profiles = list()
        final_states = list()

        for n_sim in range(self.number_simulations):
            sim_time_history = self.time_histories[n_sim]
            sim_rxn_history = self.reaction_histories[n_sim]
            sim_species_profile = dict()
            sim_rxn_profile = dict()
            state = copy.deepcopy(self.initial_state)
            for mol_ind in state:
                sim_species_profile[mol_ind] = [(0.0, self.initial_state_dict[mol_ind])]
            total_iterations = len(sim_rxn_history)

            for iter in range(total_iterations):
                rxn_ind = sim_rxn_history[iter]
                t = sim_time_history[iter]
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
