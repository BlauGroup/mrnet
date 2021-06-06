# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
from ast import literal_eval, operator
from functools import reduce
import mrnet
from mrnet.stochastic.analyze import SimulationAnalyzer, NetworkUpdater

__author__ = "Hetal D. Patel"
__maintainer__ = "Hetal D. Patel"
__email__ = "hpatel@lbl.gov"
__version__ = "0.1"
__status__ = "Alpha"
__date__ = "May, 2021"




def reaction_category_RNMC(network_folder, entriesbox, reaction_ids):

    category_dict = {"single_element_reactant":[], "redox":[],"Li_hopping":[], "Li_coord_change":[],
                     "H_or_F_abstraction":[], "actual_bond_changes": [], "non_local_reaction_center": []}
    sa = SimulationAnalyzer(network_folder, entriesbox)
    for rxn_id in reaction_ids:
        r_info = sa.index_to_reaction(rxn_id)
        reactants = []
        products = []
        for r in r_info["reactants"]:
            reactants.append(entriesbox.entries_list[r])
        for p in r_info["products"]:
            products.append(entriesbox.entries_list[p])
        r = [reactants, products]
        rxn_type = reaction_category(r)
        category_dict[rxn_type].append(rxn_id)


def update_rates_RNMC(network_folder, category_dict, rates=None):

    update = []
    if rates is None:
        rates = {"redox": 0.1, "Li_hopping": 0.1, "Li_coord_change": 0.1, "H_or_F_abstraction": 0.1}

    for rxn_type in rates:
        rate = rates[rxn_type]
        for rxn_id in category_dict[rxn_type]:
            update.append((rxn_id, rate))

    network_updater = NetworkUpdater(network_folder)
    network_updater.update_rates(update)

def reaction_category(r): #r = [[reactnat molecule entries], [product moleucle entries]]


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
            print(i.species)
            single_elem_react_or_prod = True
            if i.formula == "Li1":
                single_elemt_Li = True




    mapping_dict = mrnet.core.reactions.get_reaction_atom_mapping(r[0], r[1])
    print("reactant speices", all_reactants)
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
    print(react_global_bonds, prod_global_bonds)
    r_set = set(map(lambda x: repr(x), react_global_bonds))
    p_set = set(map(lambda x: repr(x), prod_global_bonds))
    diff_r_p = list(map(lambda y: literal_eval(y), r_set - p_set))
    diff_p_r = list(map(lambda y: literal_eval(y), p_set - r_set))
    print("diff_p-r", diff_p_r)
    print("diff_r-p", diff_r_p)

    if diff_r_p != []:
        r_p = reduce(operator.concat, diff_r_p)
    else:
        r_p = []
    if diff_p_r != []:
        p_r = reduce(operator.concat, diff_p_r)
    else:
        p_r = []

    print("!!", r_p, p_r)

    # if len(list((set(r_p) & set(p_r)))) != 0:
    # print("reaction center behaving")
    Li_ind = [i for i, x in enumerate(all_reactant_species) if x == "Li"]
    # if Li_ind != []:

    if r_charge != p_charge:
        print("redox")



    elif r_p == [] or p_r == []:
        print("either only bond forming or breaking")
        combined_diff = diff_p_r + diff_r_p
        print(combined_diff)
        if combined_diff == []:
            print("UNSURE")
        elif list(set(combined_diff[0]).intersection(*combined_diff)) == []:  # [(10,1), (10,9), (10,3)]
            print("either only forming or breaking bonds during the reaction AND not following rxn center")
            print("non_local_reaction_center")
            if LiF:
                print(
                    "LiF forming - either only forming or breaking bonds during the reaction AND not following rxn center")
                print("LiF forming - non_local_reaction_center")

        elif single_elem_react_or_prod:
            print("single element bond forming or breaking")
            if single_elemt_Li:
                print("Li coordination bond Li+A <> LiA")
            else:
                print("for AutoTS")


        elif (set(Li_ind) & set(r_p)) or (set(Li_ind) & set(p_r)):  # [10,4,5]
            # print((set(Li_ind) & set(r_p)),(set(Li_ind) & set(p_r)))
            if len(r[0]) == 1 and len(r[1]) == 1:
                print("change in corrdination within a molecule")
            elif LiF:
                print("LiF coordinating")
            else:
                print("for AutoTS")
        else:
            print("coord and covalent bond break/form")


    else:
        print("bonds breaking AND forming", single_elem_react_or_prod, Li_ind)

        if len(list((set(r_p) & set(p_r)))) == 0:  # reaction center
            # list(set(diff_r_p[0]).intersection(*diff_r_p)) == [] or list(set(diff_p_r[0]).intersection(*diff_p_r))
            # == []:
            # print("change in corrdination within a molecule")
            print("non_local_reaction_center")


        elif (set(Li_ind) & set(r_p)) and (
                set(Li_ind) & set(p_r)):  # some sort of Li bonding AND Li breaking is involved
            Li_changing = (set(Li_ind) & set(r_p) & set(p_r))
            if len(r[0]) == 1 and len(r[1]) == 1:
                print("change in corrdination within a molecule")
            elif "F" in all_reactant_species:
                F_ind = [i for i, x in enumerate(all_reactant_species) if x == "F"]

                edge = [[x, y] for x in F_ind for y in Li_changing]
                for e in edge:
                    e.sort()
                    if e in react_global_bonds and e in prod_global_bonds:
                        Li_hopping = True
                if Li_hopping:
                    print("LiF hopping")
                else:
                    print("Li_hopping - inside")
            else:
                print("Li_hopping")



        elif (set(Li_ind) & set(r_p)) or (set(Li_ind) & set(p_r)):
            if single_elemt_Li:
                # print((set(Li_ind) & set(r_p)),(set(Li_ind) & set(p_r)))
                print("coord and covalent bond break/form")
            else:
                print("for AutoTS")

        elif single_elem_react_or_prod:
            print("single element bond forming or breaking")
            if single_elemt_Li:
                print("Li coordination bond Li+A <> LiA")
            else:
                print("for AutoTS")
        else:
            print("for AutoTS")





        all_reactant_species = []
        react_global_bonds = []
        prod_global_bonds = []
        r_charge = 0
        p_charge = 0
        single_elem_reactant = False

        for i in r[0]:
            r_charge = r_charge + i.charge
            all_reactant_species = all_reactant_species + i.species
            if len(i.species) == 1:
                single_elem_reactant = True
        for i in r[1]:
            p_charge = p_charge + i.charge

        if single_elem_reactant:
            return "single_element_reactant"
        elif r_charge != p_charge:
            return "redox"
        else:
            mapping_dict = mrnet.core.reactions.get_reaction_atom_mapping(r[0], r[1])
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
            print("p-r", diff_p_r)
            print("r-p", diff_r_p)
            if diff_r_p != []:
                r_p = reduce(operator.concat, diff_r_p)
            else:
                r_p = []
            if diff_p_r != []:
                p_r = reduce(operator.concat, diff_p_r)
            else:
                p_r = []

            if r_p == [] or p_r == []:
                print("either only forming or breaking bonds during the reaction")
                if list(set(diff_r_p[0]).intersection(*diff_r_p)) == []:
                    print("either only forming or breaking bonds during the reaction AND not following rxn center")
                    return "non_local_reaction_center"
                elif len(list((set(r_p) & set(p_r)))) != 0:
                    print("reaction center behaving")
                    Li_ind = [i for i, x in enumerate(all_reactant_species) if x == "Li"]
                    if Li_ind != []:
                        if (set(Li_ind) & set(r_p)) and (set(Li_ind) & set(p_r)):
                            print("in both - so hopping")
                            return "Li_hopping"
                        elif (set(Li_ind) & set(r_p)) or (set(Li_ind) & set(p_r)):
                            print("only in one - so coord")
                            return "Li_coord_change"
                    elif "H" in all_reactant_species or "F" in all_reactant_species:
                        print("actual bond change?")
                        if len(set(r_p) & set(p_r)) == 1 and all_reactant_species[list(set(r_p) & set(p_r))[0]] == "H" or all_reactant_species[
                            list(set(r_p) & set(p_r))[0]] == "F":
                            print("H or F abstraction")
                            return "H_or_F_abstraction"
                    else:
                        return "actual_bond_changes"
                # print(r_p, p_r, list(set(r_p) & set(p_r))[0])





