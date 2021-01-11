import copy
import heapq
import itertools
import time as time
from typing import Dict, List, Tuple

import networkx as nx
from monty.json import MSONable
from networkx.readwrite import json_graph

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import (
    ConcertedReaction,
    CoordinationBondChangeReaction,
    IntermolecularReaction,
    IntramolSingleBondChangeReaction,
    Reaction,
    RedoxReaction,
    exponent,
    general_graph_rep,
    rexp,
    softplus,
)
from mrnet.utils.classes import load_class

__author__ = "Sam Blau, Hetal Patel, Xiaowei Xie, Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Sam Blau"
__status__ = "Alpha"


Mapping_Record_Dict = Dict[str, List[str]]


class ReactionPath(MSONable):
    """
    A class to define path object within the reaction network which
    constains all the associated characteristic attributes of a given path
    :param path - a list of nodes that defines a path from node A to B
        within a graph built using ReactionNetwork.build()
    """

    def __init__(self, reactions, cost, description = Optional[str]):
        """
        initializes the ReactionPath object attributes for a given path
        :param path: a list of nodes that defines a path from node A to B
            within a graph built using ReactionNetwork.build()
        """

        self.reactions = reactions
        self.cost = cost
        self.description = description

    @abstractproperty
    def overall_free_energy_change(self):
      " This method determines the overall free energy change for this reaction "
    
    @abstractproperty
    def most_uphill_step(self):
      " Determines the most uphill step in free energy in this reaction path"
   
    @abstractmethod
    def most_costly_step(self, calc: ReactionRateCalculator):
      " Determines the most costly step as determined by the cost function in the calculator "
      
      
