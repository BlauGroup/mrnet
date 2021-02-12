from abc import abstractmethod
from monty.json import MSONable
from mrnet.core.reaction import Reaction

class CostFunction(MSONable):
  " This defined an abstract interface to a reaction cost calculator which takes a reaction and computes the cost as a positive number "
  
  @abstractmethod
  def compute_cost(self, reaction: Reaction) -> float:
    """
    Computes the cost as a positive function
    Raises a ValueError if the reaction is not compatible with this calculator
    """


