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


class SerializedReactionNetwork:
    """
    An object designed to store data from a ReactionNetwork suitable for use with
    the C RNMC code.
    """

    def __init__(
        self,
        reaction_network: Union[ReactionNetwork, ReactionGenerator],
        initial_state_data: List[Tuple[MoleculeEntry, int]],
        network_folder: str,
        param_folder: str,
        logging: bool = False,
        positive_weight_coef: float = 38.61,
        all_rate_coefficients_are_one=False,
    ):

        if isinstance(reaction_network, ReactionGenerator):
            reactions = reaction_network
            entries_list = reaction_network.rn.entries_list

        else:
            reactions = reaction_network.reactions
            entries_list = reaction_network.entries_list

        self.network_folder = network_folder
        self.param_folder = param_folder
        self.logging = logging
        self.positive_weight_coef = positive_weight_coef
        self.all_rate_coefficients_are_one = all_rate_coefficients_are_one

        self.__extract_index_mappings(reactions)
        if logging:
            print("extracted index mappings")

        self.__extract_species_data(entries_list)
        if logging:
            print("extracted species data")

        self.initial_state = np.zeros(self.number_of_species)
        for (mol_entry, count) in initial_state_data:
            index = self.mol_entry_to_internal_index(mol_entry)
            self.initial_state[index] = count

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

    def visualize_molecules(self):
        folder = self.network_folder + "/molecule_diagrams"
        if os.path.isdir(folder):
            return

        os.mkdir(folder)
        for index in range(self.number_of_species):
            molecule_entry = self.species_data[index]
            visualize_molecule_entry(molecule_entry, folder + "/" + str(index) + ".pdf")

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
            if self.all_rate_coefficients_are_one:
                reaction["rate_constant"] = 1.0
            else:
                dG = reaction["free_energy"]
                if dG > 0:
                    rate = math.exp(-self.positive_weight_coef * dG)
                else:
                    rate = math.exp(-dG)
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

    def serialize(self):
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

        folder = self.network_folder

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
            f.write(("%e" % 1.0) + "\n")

        with open(folder + factor_zero_postfix, "w") as f:
            f.write(("%e" % 1.0) + "\n")

        with open(folder + factor_duplicate_postfix, "w") as f:
            f.write(("%e" % 1.0) + "\n")

        with open(folder + rates_postfix, "w") as f:
            for reaction in self.index_to_reaction:
                f.write(("%e" % reaction["rate_constant"]) + "\n")

        with open(folder + initial_state_postfix, "w") as f:
            for i in range(self.number_of_species):
                f.write(str(int(self.initial_state[i])) + "\n")

        with open(folder + "/rnsd.pickle", "wb") as f:
            pickle.dump(self, f)

        print("finished serializing")


def serialize_simulation_parameters(
    folder: str,
    number_of_threads: int,
    step_cutoff: Optional[int] = 1000000,
    time_cutoff: Optional[float] = None,
    seeds: Optional[List[int]] = None,
    number_of_simulations: Optional[int] = 100,
):
    """
    write simulation paramaters to a file so that they can be ingested by RNMC

    Args:
        folder (Path): Folder in which to store simulation parameters
        number_of_threads (int): Number of threads to use in simulation
        step_cutoff (int, or None): Number of steps to allow in each simulation.
            Default is 1,000,000
        time_cutoff (float, or None): Time duration of each simulation. Default
            is None
        seeds (List of ints, or None): Random seeds to use for simulations.
            Default is None, meaning that these seeds will be randomly
            generated.
        number_of_simulations (int, or None): Number of simulations. If seeds is
            None (default), then this number (default 100) of random seeds will
            be randomly generated.
    """

    number_of_seeds_postfix = "/number_of_seeds"
    number_of_threads_postfix = "/number_of_threads"
    seeds_postfix = "/seeds"
    time_cutoff_postfix = "/time_cutoff"
    step_cutoff_postfix = "/step_cutoff"

    if seeds is not None:
        number_of_seeds = len(seeds)
        random_seeds = seeds
    elif number_of_simulations is not None:
        number_of_seeds = number_of_simulations
        random_seeds = random.sample(list(range(1, sys.maxsize)), number_of_simulations)
    else:
        raise ValueError(
            "Need either number of simulations or set of seeds to proceed!"
        )

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
        f.write(str(number_of_seeds) + "\n")

    with open(folder + number_of_threads_postfix, "w") as f:
        f.write(str(number_of_threads) + "\n")

    with open(folder + seeds_postfix, "w") as f:
        for seed in random_seeds:
            f.write(str(seed) + "\n")


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


class SimulationAnalyser:
    """
    A class to analyze the resutls of a set of MC runs
    """

    def __init__(self, rnsd: SerializedReactionNetwork, network_folder: str):
        """
        Params:
            rnsd (SerializedReactionNetwork):
            network_folder (Path):
        """

        self.network_folder = network_folder
        self.histories_folder = network_folder + "/simulation_histories"
        self.rnsd = rnsd
        self.initial_state = rnsd.initial_state
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
        self.rnsd.visualize_molecules()

    def extract_species_consumption_info(
            self,
            target_species_index: int) -> Tuple[Dict[int, int], Dict[int, int], List[int]]:
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
            self.rnsd.network_folder
            + "/consumption_report_"
            + str(target_species_index)
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
        folder = (
            self.rnsd.network_folder + "/pathway_report_" + str(target_species_index)
        )
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

        folder = self.rnsd.network_folder + "/reaction_tally_report"
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


def run(
    molecule_entries: List[MoleculeEntry],
    initial_state: List[Tuple[MoleculeEntry, int]],
    network_folder: str,
    param_folder: str,
    number_of_threads: int = 4,
    number_of_steps: int = 200,
    number_of_simulations: int = 1000,
    all_rate_coefficients_are_one: bool = False,
) -> SimulationAnalyser:
    """
    procedure which takes a list of molecule entries + initial state and runs
    RNMC. It returns a simulation analyser which bundles together all the analysis
    procedures.
    """

    reaction_generator = ReactionGenerator(molecule_entries)
    rnsd = SerializedReactionNetwork(
        reaction_generator,
        initial_state,
        network_folder,
        param_folder,
        logging=False,
        all_rate_coefficients_are_one=all_rate_coefficients_are_one,
    )

    rnsd.serialize()

    serialize_simulation_parameters(
        rnsd.param_folder,
        number_of_threads,
        step_cutoff=number_of_steps,
        seeds=list(range(1000, 1000 + number_of_simulations + 1000)),
        number_of_simulations=number_of_simulations,
    )

    os.system("RNMC " + network_folder + " " + param_folder)

    simulation_analyzer = SimulationAnalyser(rnsd, network_folder)

    return simulation_analyzer


def resume_analysis(network_folder: str) -> SimulationAnalyser:
    """
    as part of serialization, the SerializedReactionNetwork is stored as a
    pickle in the network folder. This allows for analysis to be picked up in a
    new python session.
    """
    with open(network_folder + "/rnsd.pickle", "rb") as f:
        rnsd = pickle.load(f)

    sa = SimulationAnalyser(rnsd, network_folder)
    return sa
