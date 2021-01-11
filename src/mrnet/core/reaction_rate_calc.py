from abc import abstractmethod
from mrnet.core.reaction import Reaction

class ReactionRateCalculator:
  " This defined an abstract interface to a reaction rate calculator which takes a reaction and computes the rate constant in ** "
  
  @abstractmethod
  def compute_rate(self, reaction: Reaction) -> float:
    """
    Computes the reaction rate in units of ** 
    Raises a ValueError if the reaction is not compatibel with this calculator
    """
