from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from typing import List
from monty.json import MSONable
from mrnet.core.costs import CostFunction

class ReactionPath(MSONable, meta=ABCMeta):
    """
    A class to define path object within the reaction network
    """

    @abstractproperty
    def products(self):
      " The final products of this pathway "

    @abstractproperty
    def reactants(self):
      "  The reactants "

    @abstractproperty
    def total_energy_change(self):
      " This method determines the overall energy change for this reaction "
    
    @abstractproperty
    def most_uphill_step(self):
      " Determines the most uphill step in energy in this reaction path"
   
