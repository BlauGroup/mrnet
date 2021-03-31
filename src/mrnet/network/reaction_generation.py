import networkx as nx
from monty.json import MSONable
import itertools
import time as time
from typing import Dict, List, Tuple, Union, Any, FrozenSet, Set
from mrnet.network.reaction_network import ReactionNetwork
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import (
    ConcertedReaction,
    CoordinationBondChangeReaction,
    IntermolecularReaction,
    IntramolSingleBondChangeReaction,
    Reaction,
    RedoxReaction,
)


from mrnet.utils.classes import load_class
from multiprocessing import Pool
from functools import partial

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith, Daniel Barter"
__maintainer__ = "Daniel Barter"

def generate_concerted_reactions(
        rn,
        mols_to_keep,
        single_elem_interm_ignore,
        entry):
    return ReactionNetwork.identify_concerted_rxns_for_specific_intermediate(
            entry,
            rn,
            mols_to_keep=mols_to_keep,
            single_elem_interm_ignore=single_elem_interm_ignore,
        )


class ReactionGenerator:
    """
    takes a list of molecule entries and produces the concerted
    reactions in batches grouped by intermediate by calling
    ReactionNetwork.identify_concerted_rxns_for_specific_intermediate.
    This allows looping over concerteds without needing to have them
    all reside in memory simultaneously
    """

    def generate_concerted_reactions_parallel(
            self,
            entries: List[MoleculeEntry],
    ) -> List[ConcertedReaction]:
        with Pool(self.number_of_threads) as p:
            ls = p.map(
                partial(
                    generate_concerted_reactions,
                    self.rn,
                    [e.parameters["ind"] for e in self.rn.entries_list],
                    self.single_elem_interm_ignore),
                entries)

        reactions = []
        for l in ls:
            for c in l[0]:
                reactions.append(tuple(c))
        return_list = []

        for (reactants, products) in reactions:
            new_reactants = []
            new_products = []
            for reactant_id in reactants:
                if reactant_id is not None:
                    new_reactants.append(self.rn.entries_list[reactant_id])

            for product_id in products:
                if product_id is not None:
                    new_products.append(self.rn.entries_list[product_id])

            cs = ConcertedReaction(
                new_reactants,
                new_products,
                electron_free_energy=self.rn.electron_free_energy,
            )

            if cs:
                return_list.append(cs)
            else:
                print("concerted reaction not created:")
                print("reactants:", reactants)
                print("products:", products)

        return return_list



    def next_chunk(self):

        next_chunk = []
        while not next_chunk:
            next_indices = []
            for i in range(self.batch_size):
                j = i + self.intermediate_index
                if j < len(self.rn.entries_list):
                    next_indices.append(j)

            if len(next_indices) == 0:
                raise StopIteration()

            self.intermediate_index += len(next_indices)
            next_chunk = self.current_chunk = self.generate_concerted_reactions_parallel(
                [self.rn.entries_list[i] for i in next_indices]
            )

        self.chunk_index = 0

    def next_reaction(self):

        while True:
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
        input_entries,
        number_of_threads = 5,
        batch_size = 100,
        single_elem_interm_ignore=["C1", "H1", "O1", "Li1", "P1", "F1"],
    ):

        self.rn = ReactionNetwork.from_input_entries(input_entries)
        self.rn.build()
        self.rn.build_matrix()
        self.single_elem_interm_ignore = single_elem_interm_ignore
        self.number_of_threads = number_of_threads
        self.batch_size = batch_size

        # generator state

        self.current_chunk = self.rn.reactions
        self.chunk_index = 0
        self.intermediate_index = 0
