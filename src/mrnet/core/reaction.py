from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from typing import List, TypeVar, Protocol
from monty.json import MSONable
from mrnet.core.entry import Entry

T = TypeVar("T", bound="Reaction")


class Reaction(MSONable, metaclass=ABCMeta):
    """
    An abstract class to store info on Reactions
    """

    @abstractproperty
    def reactants(self) -> List[Entry]:
        " List of reactant Entries "

    @abstractproperty
    def products(self) -> List[Entry]:
        " List of product Entries "

    @abstractproperty
    def energy(self) -> float:
        " Energy of this reaction in eV"

    @abstractmethod
    def forward_rate(
        self,
        temperature: float = 298.0,
        electron_potential: float = 0.0,
        kappa: float = 1.0,
    ):
        """
        Compute the forward reaction rate

        Args:
            temperature: the temperature of the reaction in Kelvin
            electron_potential: the electron potential in eV
            kappa: the transmission coefficient which ranges from 0.0 to 1.0
                   and is typically 1.0 for perfect transition state theory
        """

    @abstractmethod
    def reverse_rate(
        self,
        temperature: float = 298.0,
        electron_potential: float = 0.0,
        kappa: float = 1.0,
    ):
        """
        Compute the reverse reaction rate

        Args:
            temperature: the temperature of the reaction in Kelvin
            electron_potential: the electron potential in eV
            kappa: the transmission coefficient which ranges from 0.0 to 1.0
                   and is typically 1.0 for perfect transition state theory
        """


class ReactionFactory(Protocol):
    " Basic reaction factory to generate reactions from Entries "

    @classmethod
    def from_entries(cls, entries: List[Entry]) -> List[Reaction]:
        """
        Generate reactions from the Entries

        Args:
            entries: list of entries to use
        """


class DependentReactionFactor(Protocol):
    " Reaction factory that builds reactions from already constructed reactions "

    @classmethod
    def from_reactions(cls, reactios: List[Reaction]) -> List[Reaction]:
        """
        Generate reactions from Reactions

        Args:
            entries: list of entries to use
        """