# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
from ast import literal_eval
from functools import reduce
import operator
import math
import mrnet
from mrnet.stochastic.analyze import SimulationAnalyzer, NetworkUpdater
from mrnet.utils.constants import KB, ROOM_TEMP, PLANCK
from mrnet.utils.visualization import generate_latex_header, generate_latex_footer

__author__ = "Hetal D. Patel"
__maintainer__ = "Hetal D. Patel"
__email__ = "hpatel@lbl.gov"
__version__ = "0.1"
__status__ = "Alpha"
__date__ = "May, 2021"


def reaction_category_RNMC(
    simulation_analyzer,
    entriesbox,
    reaction_ids,
    category_dict={},
    initial_mol_molentries=None,
):
    """
    Method to categorize reactions from RNMC based on reaction ids
    :param network_folder: path to network folder
    :param entriesbox: EntriesBox instance of the entries
    :param reaction_ids: list of RNMC reaction ids
    :param category_dict: if pre-existing dictionary with categorization exist
    :return: category_dict: dict with key being the type of reactions and item being
        a list of reaction ids
    """

    for rxn_id in reaction_ids:
        r_info = simulation_analyzer.index_to_reaction(rxn_id)
        reactants = []
        products = []
        for r in r_info["reactants"]:
            reactants.append(entriesbox.entries_list[r])
        for p in r_info["products"]:
            products.append(entriesbox.entries_list[p])
        r = [reactants, products]
        rxn_type = reaction_category(r, initial_mol_molentries)
        if rxn_type not in category_dict.keys():
            category_dict[rxn_type] = [rxn_id]
        else:
            category_dict[rxn_type].append(rxn_id)

    for rxn_type in category_dict:
        category_dict[rxn_type] = list(set(category_dict[rxn_type]))

    return category_dict


def dG_based_barrier_update_RNMC(
    simulation_analyzer,
    network_folder_to_update,
    reaction_ids,
    barrier=0.24,
    abs_cutoff=0.08,
):

    kT = KB * ROOM_TEMP
    max_rate = kT / PLANCK
    rate = max_rate * math.exp(-barrier / kT)
    update = []
    for rxn_id in reaction_ids:
        r_info = simulation_analyzer.index_to_reaction(rxn_id)
        dG = r_info["free_energy"]
        if abs(dG) <= abs_cutoff:
            update.append((rxn_id, rate))

    network_updater = NetworkUpdater(network_folder_to_update)
    network_updater.update_rates(update)


def reaction_extraction_from_pathway(
    simulation_analyzer, target_id, num_paths=100, sort_by="weight"
):

    simulation_analyzer.extract_reaction_pathways(target_id)
    reaction_dict = simulation_analyzer.reaction_pathways_dict[target_id]
    if reaction_dict == {}:
        return "No paths to " + str(target_id)
    else:
        paths = list(reaction_dict.keys())
        if sort_by == "weight":
            paths_sorted = sorted(
                paths,
                key=lambda x: (
                    reaction_dict[x]["weight"],
                    reaction_dict[x]["frequency"],
                ),
            )
        else:
            paths_sorted = sorted(
                paths,
                key=lambda x: (
                    reaction_dict[x]["frequency"],
                    reaction_dict[x]["weight"],
                ),
            )
        paths_sorted = [list(x) for x in paths_sorted]
        top_paths_sorted = [list(x) for x in paths_sorted][0:num_paths]
        all_reactions = list(set(reduce(operator.concat, top_paths_sorted)))

        return all_reactions


def generate_categorization_report(
    sa, reports_folder, category_dict, categories_to_print=[]
):

    if categories_to_print == []:
        categories_to_print = list(category_dict.keys())
    with open(
        reports_folder + "/categorization.tex",
        "w",
    ) as f:
        generate_latex_header(f)
        for rxn_type, reactions in category_dict.items():
            f.write("\n\n\n")
            str_type = str(
                "Reaction Type: " + rxn_type + " count: " + str(len(reactions))
            ).replace("_", " ")
            f.write(str_type)
            if rxn_type in categories_to_print:
                f.write("\n\n\n")
                for reaction_index in reactions:
                    sa.latex_emit_reaction(f, reaction_index)
        generate_latex_footer(f)


def update_rates_RNMC(network_folder_to_udpate, category_dict, barrier_dict=None):
    """
    Method to update rates of reactions based on the type of reactions and its reaction id
    :param network_folder: path to network folder
    :param category_dict: category_dict: dict with key being the type of reactions and item
    being a list of reaction ids
    :param rates: dict with key being the category of reactions to udpate barriers for
    and the value being the barrier
    value to update to
    """
    update = []
    if barrier_dict is None:
        barrier_dict = {
            "Li_hopping": 0.24,
        }
    kT = KB * ROOM_TEMP
    max_rate = kT / PLANCK
    rate_dict = {}
    for category, barrier in barrier_dict.items():
        rate_dict[category] = max_rate * math.exp(-barrier / kT)
    for category, rate in rate_dict.items():
        for rxn_id in category_dict[category]:
            update.append((rxn_id, rate))

    network_updater = NetworkUpdater(network_folder_to_udpate)
    network_updater.update_rates(update)


def supress_rxns_RNMC(network_folder_to_update, reactions_ids):
    """
    Method to add high barriers to reactions
    :param network_folder_to_update: path to network folder
    :param reactions_ids: list of reaction ids
    """

    update = []
    for r_id in reactions_ids:
        update.append((r_id, 0))

    network_updater = NetworkUpdater(network_folder_to_update)
    network_updater.update_rates(update)


def reaction_category(r, initial_mol_molentries=None):
    """
    Mehtod to categroize a single reaction
    :param r: list of reactant and product MoleculeEntry [[reactnat molecule entries],
    [product moleucle entries]]
    :return: string indicating type of reaction
    """

    all_reactants = []
    react_global_bonds = []
    prod_global_bonds = []
    r_charge = 0
    p_charge = 0
    single_elem_react_or_prod = False
    single_elemt_Li = False
    LiF = False
    Li_hopping = False

    for i in r[0]:
        r_charge = r_charge + i.charge
        all_reactants = all_reactants + i.species
        if i.formula == "F1 Li1":
            LiF = True
        elif len(i.species) == 1:
            single_elem_react_or_prod = True
            if i.formula == "Li1":
                single_elemt_Li = True
    for i in r[1]:
        p_charge = p_charge + i.charge
        if i.formula == "F1 Li1":
            LiF = True
        elif len(i.species) == 1:
            single_elem_react_or_prod = True
            if i.formula == "Li1":
                single_elemt_Li = True

    mapping_dict = mrnet.core.reactions.get_reaction_atom_mapping(r[0], r[1])
    all_reactant_species = all_reactants
    for r_id, react in enumerate(r[0]):
        for b in react.bonds:
            s = (mapping_dict[0][r_id][b[0]], mapping_dict[0][r_id][b[1]])
            s = sorted(s)
            react_global_bonds.append(s)
    for p_id, prod in enumerate(r[1]):
        for b in prod.bonds:
            s = (mapping_dict[1][p_id][b[0]], mapping_dict[1][p_id][b[1]])
            s = sorted(s)
            prod_global_bonds.append(s)
    r_set = set(map(lambda x: repr(x), react_global_bonds))
    p_set = set(map(lambda x: repr(x), prod_global_bonds))
    diff_r_p = list(map(lambda y: literal_eval(y), r_set - p_set))
    diff_p_r = list(map(lambda y: literal_eval(y), p_set - r_set))

    if diff_r_p != []:
        r_p = reduce(operator.concat, diff_r_p)
    else:
        r_p = []
    if diff_p_r != []:
        p_r = reduce(operator.concat, diff_p_r)
    else:
        p_r = []

    Li_ind = [i for i, x in enumerate(all_reactant_species) if x == "Li"]

    if r_charge != p_charge:  # redox
        return "redox"

    elif r_p == [] or p_r == []:  # bonds are either only forming or only breaking
        combined_diff = diff_p_r + diff_r_p
        if (
            combined_diff == []
        ):  # there is no difference in reactant and product graphs, so change transfer maybe
            # occuring
            return "uncategorized"
        elif (
            list(set(combined_diff[0]).intersection(*combined_diff)) == []
        ):  # reaction center rule not satisfied
            if LiF:
                return "non_local_reaction_center_LiF_forming"
            else:
                return "non_local_reaction_center"

        elif single_elem_react_or_prod:
            if single_elemt_Li:
                return "Li_coordination_Li+A_to_LiA"
            else:
                return "AutoTS"

        elif (set(Li_ind) & set(r_p)) or (
            set(Li_ind) & set(p_r)
        ):  # bond changes involve Li
            if len(r[0]) == 1 and len(r[1]) == 1:
                return "coordination_change_within_molecule"  # EC monodentate <> EC bidentate
            elif LiF:
                return "LiF_coordinating"  # A + LiF <> ALiF
            else:
                return "AutoTS"
        else:
            return "AutoTS"

    else:  # bonds are being broken AND formed
        if len(list((set(r_p) & set(p_r)))) == 0:  # check for reaction center
            return "non_local_reaction_center"

        elif (set(Li_ind) & set(r_p)) and (
            set(Li_ind) & set(p_r)
        ):  # some sort of Li bonding AND Li breaking
            Li_changing = set(Li_ind) & set(r_p) & set(p_r)
            if len(r[0]) == 1 and len(r[1]) == 1:
                return "coordination_change_within_molecule"
            elif "F" in all_reactant_species:
                F_ind = [i for i, x in enumerate(all_reactant_species) if x == "F"]
                edge = [[x, y] for x in F_ind for y in Li_changing]
                for e in edge:
                    e.sort()
                    if e in react_global_bonds and e in prod_global_bonds:
                        Li_hopping = True
                if Li_hopping:
                    return "LiF_hopping"  # ALiF + B <> A + BLiF
                else:
                    if initial_mol_molentries is not None:
                        if (
                            set(r[0]).intersection(set(initial_mol_molentries)) != set()
                            and set(r[1]).intersection(set(initial_mol_molentries))
                            != set()
                        ):
                            return "Li_hopping_initial_mol_in_reactant_and_product"
                        else:
                            return "Li_hopping"  # LiA + B <> A + LiB
                    else:
                        return "Li_hopping"  # LiA + B <> A + LiB
            else:
                if initial_mol_molentries is not None:
                    if (
                        set(r[0]).intersection(set(initial_mol_molentries)) != set()
                        and set(r[1]).intersection(set(initial_mol_molentries)) != set()
                    ):
                        return "Li_hopping_initial_mol_in_reactant_and_product"
                    else:
                        return "Li_hopping"  # LiA + B <> A + LiB
                else:
                    return "Li_hopping"  # LiA + B <> A + LiB

        elif (set(Li_ind) & set(r_p)) or (
            set(Li_ind) & set(p_r)
        ):  # some sort of Li bonding OR breaking
            if single_elemt_Li:
                return "coord_and_covalent_bond_changes"  # ex. Li coordination
                # causes covalent bond breakage
            else:
                return "AutoTS"

        elif single_elem_react_or_prod:
            if single_elemt_Li:
                return "Li_coordination_Li+A_to_LiA"
            else:
                return "AutoTS"
        else:
            return "AutoTS"
