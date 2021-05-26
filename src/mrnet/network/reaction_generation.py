import networkx as nx
from monty.json import MSONable
import itertools
import time as time
from typing import Dict, List, Tuple, Union, Any, FrozenSet, Set
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import (
    ConcertedReaction,
    CoordinationBondChangeReaction,
    IntermolecularReaction,
    IntramolSingleBondChangeReaction,
    Reaction,
    RedoxReaction,
)
import copy


from mrnet.utils.classes import load_class


__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith, Daniel Barter"
__maintainer__ = "Daniel Barter"


metals = ["Li", "Na", "K", "Mg", "Ca", "Zn", "Al"]
m_formulas = [m + "1" for m in metals]


class EntriesBox:
    """
    function for preprocessing a list of molecule centries. In particular, they get sorted
    by features in the attribute entries_dict and given fixed explicit indices
    """

    def __init__(
        self, input_entries, temperature=298.15, build_dict_and_filter=True, remove_complexes=True
    ):
        print(len(input_entries), "input entries")

        # Sequential indices are essential for kMC - DO NOT CHANGE!!!
        # Sequential indices are essential for kMC - DO NOT CHANGE!!!
        # Sequential indices are essential for kMC - DO NOT CHANGE!!!
        for ii, entry in enumerate(input_entries):
            entry.parameters["ind"] = ii
        self.entries_list = input_entries
        # Sequential indices are essential for kMC - DO NOT CHANGE!!!
        # Sequential indices are essential for kMC - DO NOT CHANGE!!!
        # Sequential indices are essential for kMC - DO NOT CHANGE!!!

        if build_dict_and_filter:
            entries_dict = dict()
            filtered_entries_list = list()

            # Filter out unconnected entries, aka those that contain distinctly
            # separate molecules which are not connected via a bond
            print("Removing unconnected entries...")
            connected_entries = list()
            num_unconnected = 0
            for entry in input_entries:
                if len(entry.molecule) > 1:
                    if nx.is_weakly_connected(entry.graph):
                        connected_entries.append(entry)
                    else:
                        num_unconnected += 1
                else:
                    connected_entries.append(entry)
            print("Removed {} unconnected entries".format(num_unconnected))
            # print(len(connected_entries), "connected entries")
            assert len(input_entries) - num_unconnected == len(connected_entries)

            if remove_complexes:
                print("Removing metal-centered complex entries...")
                orig_len_connected_entries = copy.deepcopy(len(connected_entries))
                complexes = list()
                for ii, entry in enumerate(connected_entries):
                    if entry.formula not in m_formulas:
                        if any([x in entry.formula for x in metals]):
                            m_inds = [
                                i for i, x in enumerate(entry.species) if x in metals
                            ]
                            e = copy.deepcopy(entry)
                            e.mol_graph.remove_nodes(m_inds)
                            if not nx.is_weakly_connected(e.mol_graph.graph):
                                complexes.append(ii)

                print("Removed {} metal-centered complexes".format(len(complexes)))
                connected_entries = [
                    e for i, e in enumerate(connected_entries) if i not in complexes
                ]
                assert orig_len_connected_entries - len(complexes) == len(
                    connected_entries
                )

            def get_formula(x):
                return x.formula

            def get_num_bonds(x):
                return x.num_bonds

            def get_charge(x):
                return x.charge

            def get_free_energy(x):
                return x.get_free_energy(temperature=temperature)

            print(
                "Building entries_dict and filtering by isomorphism via lowest free energy..."
            )
            num_isomorphic = 0
            # Sort by formula
            sorted_entries_0 = sorted(connected_entries, key=get_formula)
            for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
                sorted_entries_1 = sorted(list(g1), key=get_num_bonds)
                entries_dict[k1] = dict()
                # Sort by number of bonds
                for k2, g2 in itertools.groupby(sorted_entries_1, get_num_bonds):
                    sorted_entries_2 = sorted(list(g2), key=get_charge)
                    entries_dict[k1][k2] = dict()
                    # Sort by charge
                    for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                        sorted_entries_3 = sorted(list(g3), key=get_free_energy)
                        if len(sorted_entries_3) > 1:
                            unique = list()
                            for entry in sorted_entries_3:
                                isomorphic_found = False
                                # Sort by graph isomorphism, taking the isomorphic
                                # entry with the lowest free energy
                                for ii, Uentry in enumerate(unique):
                                    if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                        isomorphic_found = True
                                        num_isomorphic += 1
                                        if (
                                            entry.get_free_energy() is not None
                                            and Uentry.get_free_energy() is not None
                                        ):
                                            if entry.get_free_energy(
                                                temperature
                                            ) < Uentry.get_free_energy(temperature):
                                                unique[ii] = entry
                                        elif entry.get_free_energy() is not None:
                                            unique[ii] = entry
                                        elif entry.energy < Uentry.energy:
                                            unique[ii] = entry
                                        break
                                if not isomorphic_found:
                                    unique.append(entry)
                            entries_dict[k1][k2][k3] = unique
                        else:
                            entries_dict[k1][k2][k3] = sorted_entries_3
                        for entry in entries_dict[k1][k2][k3]:
                            filtered_entries_list.append(entry)

            assert len(filtered_entries_list) + num_isomorphic == len(connected_entries)

            print(len(filtered_entries_list), "unique filtered entries")

            entry_dict_count = 0
            for formula in entries_dict:
                for num_bonds in entries_dict[formula]:
                    for charge in entries_dict[formula][num_bonds]:
                        for entry in entries_dict[formula][num_bonds][charge]:
                            entry_dict_count += 1
                            assert entry in self.entries_list
                            assert "ind" in entry.parameters
            assert entry_dict_count == len(filtered_entries_list)

            self.entries_dict = entries_dict
            self.filtered_entries_list = sorted(filtered_entries_list, key=lambda x: x.parameters["ind"])
        else:
            self.entries_dict = {}
            self.filtered_entries_list = []


class ReactionGenerator(MSONable):
    """
    Class to build a reaction network from entries
    """

    def __init__(
        self,
        entries_box,
        electron_free_energy=-2.15,
        temperature=298.15,
        solvent_dielectric=18.5,
        solvent_refractive_index=1.415,
        filter_concerted_metal_coordination=False,
        replace_ind=True,
    ):
        """
        Generate a ReactionNetwork from a set of MoleculeEntries.

        :param input_entries: list of MoleculeEntries which will make up the
            network
        :param electron_free_energy: float representing the Gibbs free energy
            required to add an electron (in eV)
        :param temperature: Temperature of the system, used for free energy
            and rate constants (in K)
        :param solvent_dielectric: Dielectric constant of the solvent medium
        :param solvent_refractive_index: Refractive index of the solvent medium
        :param replace_ind: True if reindex the entries if it there is already
            indices in the input_entries
        :param filters_metal_coordination: Remove concerted reactions involving
            metal coordination.
        :return:
        """

        self.entries_box = entries_box

        self.electron_free_energy = electron_free_energy
        self.temperature = temperature
        self.solvent_dielectric = solvent_dielectric
        self.solvent_refractive_index = solvent_refractive_index

        self.entry_ids = {e.entry_id for e in self.entries_box.entries_list}
        self.matrix = None
        self.matrix_inverse = None

        self.index_formula_mapping = {
            e.parameters["ind"]: e.formula for e in self.entries_box.entries_list
        }

        self.filters = list()
        if filter_concerted_metal_coordination:
            self.filters.append("metal_coordination")

    def build(
        self,
        reaction_types: Union[Set, FrozenSet] = frozenset(
            {
                "RedoxReaction",
                "IntramolSingleBondChangeReaction",
                "IntermolecularReaction",
                "CoordinationBondChangeReaction",
            }
        ),
        determine_atom_mappings: bool = False,
    ) -> nx.DiGraph:
        """
            A method to build the reaction network graph

        :param reaction_types (set/frozenset): set/frozenset of all the reactions
            class to include while building the graph
        :param determine_atom_mappings (bool): If True (default), create an atom
            mapping between reactants and products in a given reaction
        :return: nx.DiGraph
        """

        print("build() start", time.time())

        self.graph = nx.DiGraph()

        # Add molecule nodes
        for entry in self.entries_box.entries_list:
            self.graph.add_node(entry.parameters["ind"], bipartite=0)

        reaction_classes = [load_class(str(self.__module__), s) for s in reaction_types]

        all_reactions = list()

        # Generate reactions
        for r in reaction_classes:
            reactions = r.generate(
                self.entries_box.entries_dict,
                determine_atom_mappings=determine_atom_mappings,
            )  # review
            all_reactions.append(reactions)

        all_reactions = [i for i in all_reactions if i]
        self.reactions = list(itertools.chain.from_iterable(all_reactions))

        self.redox_c = 0
        self.inter_c = 0
        self.intra_c = 0
        self.coord_c = 0

        for ii, r in enumerate(self.reactions):
            r.parameters["ind"] = ii
            if r.__class__.__name__ == "RedoxReaction":
                self.redox_c += 1
                r.electron_free_energy = self.electron_free_energy
                r.set_free_energy()
                r.set_rate_constant()
            elif r.__class__.__name__ == "IntramolSingleBondChangeReaction":
                self.intra_c += 1
            elif r.__class__.__name__ == "IntermolecularReaction":
                self.inter_c += 1
            elif r.__class__.__name__ == "CoordinationBondChangeReaction":
                self.coord_c += 1
            self.add_reaction(r.graph_representation())  # add graph element here

        print(len(self.graph.nodes), "nodes in the graph")
        print(len(self.graph.edges), "edges in the graph")

        print(
            "redox: ",
            self.redox_c,
            "inter: ",
            self.inter_c,
            "intra: ",
            self.intra_c,
            "coord: ",
            self.coord_c,
        )

        self.build_matrix()

        print("build() end", time.time())

    def add_reaction(self, graph_representation: nx.DiGraph):
        """
            A method to add a single reaction to the ReactionNetwork.graph
            attribute
        :param graph_representation: Graph representation of a reaction,
            obtained from ReactionClass.graph_representation
        """
        self.graph.add_nodes_from(graph_representation.nodes(data=True))
        self.graph.add_edges_from(graph_representation.edges(data=True))

    def build_matrix(self) -> Dict[int, Dict[int, List[Tuple]]]:
        """
        A method to build a spare adjacency matrix using dictionaries.
        :return: nested dictionary {r1:{c1:[],c2:[]}, r2:{c1:[],c2:[]}}
        """
        self.matrix = {}
        for i in range(len(self.entries_box.entries_list)):
            self.matrix[i] = {}
        for node in self.graph.nodes:
            if isinstance(node, str):
                if "electron" not in self.graph.nodes[node]["rxn_type"]:
                    in_node = list(self.graph.predecessors(node))
                    out_nodes = list(self.graph.successors(node))
                    edges = []
                    for u in in_node:
                        for v in out_nodes:
                            edges.append((u, v))
                    for e in edges:
                        if e[1] not in self.matrix[e[0]].keys():
                            self.matrix[e[0]][e[1]] = [
                                (node, self.graph.nodes[node]["free_energy"], "e")
                            ]
                        else:
                            self.matrix[e[0]][e[1]].append(
                                (node, self.graph.nodes[node]["free_energy"], "e")
                            )
        self.matrix_inverse = {}
        for i in range(len(self.matrix)):
            self.matrix_inverse[i] = {}
            for k, v in self.matrix.items():
                if i in v.keys():
                    self.matrix_inverse[i][k] = v[i]

        return self.matrix

    @staticmethod
    def generate_node_string(combined_reactants, combined_products):
        """
        A method to genrate a reaction node string from given reactants and products.
        :param combined_reactants: list of reactant node indices, ex [1,2]
        :param combined_products: list of product node indices, ex [3,4]
        :return: node_str: string of reaction as it would be for a reaction node, ex  "1+2,3+4"
        """
        combined_reactants = list(map(str, combined_reactants))
        node_str = (
            "+".join(list(map(str, combined_reactants)))
            + ","
            + "+".join(list(map(str, combined_products)))
        )
        return node_str

    @staticmethod
    def add_reactions_to_matrix(matrix, reaction):
        """
        A method to add new concerted reactions to the matrix which is already built from elemetary reactions.
        :param matrix: matrix which is already built from elemetary reactions, self.matrix
        :param reaction: concerted reaction to add to the matrix, (1,2], [3,4], total_dG)
        :return: matrix updated with the reaction
        """
        nstr = ReactionGenerator.generate_node_string(reaction[0], reaction[1])
        for reac in reaction[0]:
            for prod in reaction[1]:
                if prod not in matrix[reac].keys():
                    matrix[reac][prod] = [(nstr, reaction[2], "c")]
                else:
                    matrix[reac][prod].append((nstr, reaction[2], "c"))
        return matrix

    def identify_concerted_rxns_for_specific_intermediate(
        self,
        entry: MoleculeEntry,
        mols_to_keep=None,
        single_elem_interm_ignore=["C1", "H1", "O1", "Li1", "P1", "F1"],
        update_matrix=False,
    ):

        """
            A method to identify concerted reactions via specific high enery intermediate molecule
        :param entry: MoleculeEntry to act as high energy intermediate
        :param RN: Reaction network built
        :param mols_to_keep: List of pruned molecules, if not running then a list of all molecule nodes
        in the RN_pr_solved
        :param single_elem_interm_ignore: single_elem_interm_ignore: List of formula of high energy
        intermediates to ignore
        :return: list of reactions
        """

        rxn_from_filer_iter1 = []
        rxn_from_filer_iter1_nodes = []
        entry_ind = entry.parameters["ind"]  # type: int

        if mols_to_keep is None:
            mols_to_keep = list(range(0, len(self.entries_box.entries_list)))
        not_wanted_formula = single_elem_interm_ignore

        if (
            entry.formula not in not_wanted_formula
            and entry.parameters["ind"] in mols_to_keep
        ):

            if update_matrix:
                self.matrix2 = copy.deepcopy(self.matrix)

            row = self.matrix[entry_ind]  # type: ignore
            col = self.matrix_inverse[entry_ind]

            for kr, vr in row.items():
                for kc, vc in col.items():
                    if kr != kc:
                        for s2 in vr:
                            for e2 in vc:
                                incoming_reaction_dG = e2[1]
                                total_dG = s2[1] + e2[1]
                                if incoming_reaction_dG > 0 and total_dG < 0:
                                    (
                                        rxn1,
                                        rxn1_nodes,
                                    ) = self.concerted_reaction_filter(e2[0], s2[0])
                                    if rxn1 is not None:
                                        rxn_from_filer_iter1.append(rxn1)
                                        rxn_from_filer_iter1_nodes.append(rxn1_nodes)
                                        if update_matrix:
                                            reaction = (rxn1[0], rxn1[1], total_dG)
                                            ReactionGenerator.add_reactions_to_matrix(
                                                self.matrix2, reaction
                                            )

        return rxn_from_filer_iter1, rxn_from_filer_iter1_nodes

    def concerted_reaction_filter(self, in_reaction_node, out_reaction_node):
        """
        A method to identify a valid concerted reaction based on stiochomtery of maximum of 2 reactants and products
        :param in_reaction_node: incoming reaction node "6,2+7"
        :param out_reaction_node: outgoing reaction node "2+1,3"
        :return: r: combined reaction's reactant and product [[1,6],[3,7]], r_node: combined reaction's reactant and
        product as well as the two reaction nodes - ex [[1,6],[3,7], "6,2+7","2+1,3"]
        """
        r = None
        r_node = None
        (
            in_reactants,
            in_products,
        ) = ReactionGenerator.parse_reaction_node(in_reaction_node)
        (out_reactants, out_products) = ReactionGenerator.parse_reaction_node(
            out_reaction_node
        )
        combined_reactants = in_reactants + out_reactants
        combined_products = in_products + out_products
        combined_reactants.sort()
        combined_products.sort()
        inter = set(combined_reactants).intersection(set(combined_products))
        for i in inter:
            combined_reactants.remove(i)
            combined_products.remove(i)
        inter = set(combined_reactants).intersection(set(combined_products))
        for i in inter:
            combined_reactants.remove(i)
            combined_products.remove(i)

        reactant_entries = [self.index_formula_mapping[e] for e in combined_reactants]
        product_entries = [self.index_formula_mapping[e] for e in combined_products]
        if 0 < len(combined_reactants) <= 2 and 0 < len(combined_products) <= 2:
            # Filter to remove concerted reactions involving metal coordination
            problem_metal = False
            if "metal_coordination" in self.filters:
                if any([e in m_formulas for e in reactant_entries]):
                    this_m_formula = [e for e in reactant_entries if e in m_formulas]
                    # Metal coordination can only be part of a concerted reaction if the same metal decoordinates
                    if not any([e in this_m_formula for e in product_entries]):
                        problem_metal = True
                        print(
                            "FILTER FAILED",
                            [e for e in reactant_entries],
                            [e for e in product_entries],
                        )
                elif any([e in m_formulas for e in product_entries]):
                    problem_metal = True
                    print(
                        "FILTER FAILED",
                        [e for e in reactant_entries],
                        [e for e in product_entries],
                    )

            if not problem_metal:
                r = [combined_reactants, combined_products]
                r_node = [
                    combined_reactants,
                    combined_products,
                    [in_reaction_node, out_reaction_node],
                ]

        return r, r_node

    @staticmethod
    def parse_reaction_node(node: str):
        """
        A method to identify reactants, PR, and prodcuts from a given reaction node string.
        :param node: string, ex. "1+2,3+4"
        :return: react_list: reactant list, ex [1,2]
        :return: prod_list: product list, ex [3,4]
        """
        react_list_str = node.split(",")[0].split("+")
        prod_list_str = node.split(",")[1].split("+")
        prod_list_str.sort()
        react_list: List[int] = [int(r) for r in react_list_str]
        prod_list: List[int] = [int(p) for p in prod_list_str]
        return (react_list, prod_list)


class ReactionIterator:
    """
    takes a list of molecule entries and produces the concerted
    reactions in batches grouped by intermediate by calling
    ReactionNetwork.identify_concerted_rxns_for_specific_intermediate.
    This allows looping over concerteds without needing to have them
    all reside in memory simultaneously
    """

    def generate_concerted_reactions(
        self,
        entry: MoleculeEntry,
    ):
        """
        generate all the concerted reactions with intermediate mol_entry
        """
        (reactions, _,) = self.rn.identify_concerted_rxns_for_specific_intermediate(
            entry,
            mols_to_keep=[e.parameters["ind"] for e in self.entries_box.entries_list],
            single_elem_interm_ignore=self.single_elem_interm_ignore,
        )

        return_list = []

        for (reactants, products) in reactions:
            new_reactants = []
            new_products = []
            for reactant_id in reactants:
                if reactant_id is not None:
                    new_reactants.append(self.entries_box.entries_list[reactant_id])

            for product_id in products:
                if product_id is not None:
                    new_products.append(self.entries_box.entries_list[product_id])

            free_energy_forward = 0.0
            for product in new_products:
                free_energy_forward += product.get_free_energy(
                    temperature=self.rn.temperature
                )

            for reactant in new_reactants:
                free_energy_forward -= reactant.get_free_energy(
                    temperature=self.rn.temperature
                )

            return_list.append((tuple(reactants), tuple(products), free_energy_forward))

        return return_list

    def next_chunk(self):

        next_chunk = []
        while not next_chunk:
            self.intermediate_index += 1

            if self.intermediate_index == len(self.entries_box.entries_list):
                raise StopIteration()

            next_chunk = self.current_chunk = self.generate_concerted_reactions(
                self.entries_box.entries_list[self.intermediate_index]
            )

            print(
                "concerted chunk for intermediate",
                self.intermediate_index,
                ">",
                len(next_chunk),
            )

        self.chunk_index = 0

    def next_reaction(self):

        if self.chunk_index == len(self.current_chunk):
            self.next_chunk()

        reaction = self.current_chunk[self.chunk_index]
        self.chunk_index += 1
        return reaction

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_reaction()

    def __init__(
        self,
        entries_box,
        electron_free_energy=-2.15,
        temperature=298.15,
        solvent_dielectric=18.5,
        solvent_refractive_index=1.415,
        single_elem_interm_ignore=["C1", "H1", "O1", "Li1", "P1", "F1"],
        filter_concerted_metal_coordination=False,
    ):

        self.entries_box = entries_box

        self.rn = ReactionGenerator(
            entries_box,
            electron_free_energy=electron_free_energy,
            temperature=temperature,
            solvent_dielectric=solvent_dielectric,
            solvent_refractive_index=solvent_refractive_index,
            filter_concerted_metal_coordination=filter_concerted_metal_coordination,
        )
        self.rn.build()
        self.single_elem_interm_ignore = single_elem_interm_ignore

        # generator state

        first_chunk_reaction_objects = self.rn.reactions
        first_chunk = [
            (
                tuple([int(r) for r in reaction.reactant_indices]),
                tuple([int(r) for r in reaction.product_indices]),
                reaction.free_energy_A,
            )
            for reaction in first_chunk_reaction_objects
        ]

        self.current_chunk = first_chunk
        self.chunk_index = 0
        self.intermediate_index = -1
