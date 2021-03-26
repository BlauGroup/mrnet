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


__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith, Daniel Barter"
__maintainer__ = "Daniel Barter"


class ReactionGenerator:
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
    ) -> List[ConcertedReaction]:
        """
        generate all the concerted reactions with intermediate mol_entry
        """
        (
            reactions,
            _,
        ) = ReactionNetwork.identify_concerted_rxns_for_specific_intermediate(
            entry, self.rn, [e.parameters["ind"] for e in self.rn.entries_list]
        )

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
            self.intermediate_index += 1

            if self.intermediate_index == len(self.rn.entries_list):
                raise StopIteration()

            next_chunk = self.current_chunk = self.generate_concerted_reactions(
                self.rn.entries_list[self.intermediate_index]
            )

        self.chunk_index = 0

    def next_reaction(self):

        while True:
            if self.chunk_index == len(self.current_chunk):
                self.next_chunk()

            reaction = self.current_chunk[self.chunk_index]
            self.chunk_index += 1
            reaction_sig = (
                frozenset(reaction.reactant_indices),
                frozenset(reaction.product_indices),
            )

            if reaction_sig not in self.previously_seen_reactions:
                self.previously_seen_reactions.add(reaction_sig)
                return reaction

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_reaction()

    def __init__(self, input_entries):

        self.rn = ReactionNetwork.from_input_entries(input_entries)
        self.rn.build()

        # generator state

        self.current_chunk = self.rn.reactions
        self.chunk_index = 0
        self.intermediate_index = -1
        self.previously_seen_reactions = set()
