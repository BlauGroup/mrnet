import logging

import numpy as np
from scipy.constants import h, k, N_A, pi, epsilon_0, elementary_charge

from monty.json import MSONable

from pymatgen.core.units import amu_to_kg

from mrnet.utils.math import product
from mrnet.utils.constants import ROOM_TEMP, KB, PLANCK


__author__ = "Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "September 2019"

logger = logging.getLogger(__name__)


class ReactionRateCalculator(MSONable):

    """
    An object which represents a chemical reaction (in terms of reactants, transition state,
    and products) and which can, from the energetics of those individual molecules, predict the
    rate constant, rate law, and thus the chemical kinetics of the reaction.

    NOTE: It is assumed that only one transition state is present.

    Args:
        reactants (list): list of MoleculeEntry objects
        products (list): list of MoleculeEntry objects
        transition_state (MoleculeEntry): MoleculeEntry representing the transition state between
            the reactants and the products

    Returns:
        None
    """

    def __init__(self, reactants, products, transition_state):

        # Assume rate law is first-order in terms of each reactant/product
        self.rate_law = {
            "reactants": np.ones(len(reactants)),
            "products": np.ones(len(products)),
        }

        # Store relevant information from reactants, products, rather than storing obj
        # Here, they are stored as np arrays, which have a smaller footprint
        self.product_energy = sum([p.energy if p.energy else 0 for p in products])
        self.reactant_energy = sum([r.energy if r.energy else 0 for r in reactants])
        self.transition_state_energy = (
            None if transition_state is None else transition_state.energy
        )

        self.product_enthalpy = sum([p.enthalpy if p.enthalpy else 0 for p in products])
        self.reactant_enthalpy = sum(
            [r.enthalpy if r.enthalpy else 0 for r in reactants]
        )
        self.transition_state_enthalpy = (
            None if transition_state is None else transition_state.enthalpy
        )

        self.product_entropy = sum([p.entropy if p.entropy else 0 for p in products])
        self.reactant_entropy = sum([r.entropy if r.entropy else 0 for r in reactants])
        self.transition_state_entropy = (
            None if transition_state is None else transition_state.entropy
        )

        self.reactant_str = " + ".join(
            [r.molecule.composition.alphabetical_formula for r in reactants]
        )
        self.product_str = " + ".join(
            [p.molecule.composition.alphabetical_formula for p in products]
        )

        # Compute net values for the reaction
        self.net_energy = (self.product_energy - self.reactant_energy) * 27.2116
        self.net_enthalpy = (self.product_enthalpy - self.reactant_enthalpy) * 0.0433641
        self.net_entropy = (self.product_entropy - self.reactant_entropy) * 0.0000433641

    def calculate_net_gibbs(self, temperature=ROOM_TEMP):
        """
        Calculate net reaction Gibbs free energy at a given temperature.

        ΔG = ΔH - T ΔS

        Args:
            temperature (float): absolute temperature in Kelvin

        Returns:
            float: net Gibbs free energy (in eV)
        """
        rct_gibbs = (
            (self.reactant_energy * 27.21139)
            + (0.0433641 * self.reactant_enthalpy)
            - (temperature * self.reactant_entropy * 0.0000433641)
        )
        pro_gibbs = (
            (self.product_energy * 27.21139)
            + (0.0433641 * self.product_enthalpy)
            - (temperature * self.product_entropy * 0.0000433641)
        )

        return pro_gibbs - rct_gibbs

    def calculate_net_thermo(self, temperature=ROOM_TEMP):
        """
        Calculate net energy, enthalpy, and entropy for the reaction.
        Args:
            temperature (float): absolute temperature in Kelvin (default 300.0K)
        Returns:
            net_thermo: dict with relevant net thermodynamic variables
        """

        net_thermo = {
            "energy": self.net_energy,
            "enthalpy": self.net_enthalpy,
            "entropy": self.net_entropy,
            "gibbs": self.calculate_net_gibbs(temperature),
        }

        return net_thermo

    def calculate_act_energy(self, reverse=False):
        """
        Calculate energy of activation.

        Args:
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            float: energy of activation (in eV)

        """
        trans_energy = self.transition_state_energy

        if reverse:
            return (trans_energy - self.product_energy) * 27.2116
        else:
            return (trans_energy - self.reactant_energy) * 27.2116

    def calculate_act_enthalpy(self, reverse=False):
        """
        Calculate enthalpy of activation.

        Args:
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            float: enthalpy of activation (in eV)

        """

        trans_enthalpy = self.transition_state_enthalpy

        if reverse:
            return (trans_enthalpy - self.product_enthalpy) * 0.0433641
        else:
            return (trans_enthalpy - self.reactant_enthalpy) * 0.0433641

    def calculate_act_entropy(self, reverse=False):
        """
        Calculate entropy of activation.

        Args:
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            float: entropy of activation (in eV/K)

        """

        trans_entropy = self.transition_state_entropy

        if reverse:
            return (trans_entropy - self.product_entropy) * 0.0000433641
        else:
            return (trans_entropy - self.reactant_entropy) * 0.0000433641

    def calculate_act_gibbs(self, temperature=ROOM_TEMP, reverse=False):
        """
        Calculate Gibbs free energy of activation at a given temperature.

        ΔG = ΔH - T ΔS

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            float: Gibbs free energy of activation (in eV)
        """

        act_energy = self.calculate_act_energy(reverse=reverse)
        act_enthalpy = self.calculate_act_enthalpy(reverse=reverse)
        act_entropy = self.calculate_act_entropy(reverse=reverse)

        return act_energy + act_enthalpy - temperature * act_entropy

    def calculate_act_thermo(self, temperature=ROOM_TEMP, reverse=False):
        """
        Calculate thermodynamics of activation for the reaction.

        Args:
            temperature (float): absolute temperature in Kelvin (default 300.0K)
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            act_thermo: dict with relevant activation thermodynamic variables
        """

        act_thermo = {
            "energy": self.calculate_act_energy(reverse=reverse),
            "enthalpy": self.calculate_act_enthalpy(reverse=reverse),
            "entropy": self.calculate_act_entropy(reverse=reverse),
            "gibbs": self.calculate_act_gibbs(temperature, reverse=reverse),
        }

        return act_thermo

    def calculate_rate_constant(self, temperature=ROOM_TEMP, reverse=False, kappa=1.0):
        """
        Calculate the rate constant k by the Eyring-Polanyi equation of transition state theory.

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction
            kappa (float): transmission coefficient (by default, we assume the assumptions of
                transition-state theory are valid, so kappa = 1.0

        Returns:
            k_rate (float): temperature-dependent rate constant
        """

        gibbs = self.calculate_act_gibbs(temperature=temperature, reverse=reverse)

        k_rate = kappa * KB * temperature / PLANCK * np.exp(-gibbs / (KB * temperature))
        return k_rate

    def calculate_rate(
        self, concentrations, temperature=ROOM_TEMP, reverse=False, kappa=1.0
    ):
        """
        Calculate the based on the reaction stoichiometry.

        NOTE: Here, we assume that the reaction is an elementary step.

        Args:
            concentrations (list): concentrations of reactant molecules (product molecules, if
                reverse=True). Order of the reactants/products DOES matter.
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction
            kappa (float): transmission coefficient (by default, we assume the assumptions of
                transition-state theory are valid, so kappa = 1)

        Returns:
            rate (float): reaction rate, based on the stoichiometric rate law and the rate constant
        """

        rate_constant = self.calculate_rate_constant(
            temperature=temperature, reverse=reverse, kappa=kappa
        )

        if reverse:
            exponents = self.rate_law["products"]
        else:
            exponents = self.rate_law["reactants"]

        rate = rate_constant * product(np.array(concentrations) ** exponents)

        return rate

    def __repr__(self):
        return "Rate Calculator for: {} --> {}".format(
            self.reactant_str, self.product_str
        )

    def __str__(self):
        return self.__repr__()


class BEPRateCalculator(ReactionRateCalculator):
    """
    A modified reaction rate calculator that uses the Bell-Evans-Polanyi principle to predict the
    activation energies (and, thus, the rate constants and reaction rates) of chemical reactions.

    The Bell-Evans-Polanyi principle states that, for reactions within a similar class or family,
    the difference in activation energy between the reactions is proportional to the difference in
    reaction enthalpy. That is,

    E_a = E_a,0 + alpha * ΔH, where

    E_a = the activation energy of the reaction of interest
    E_a,0 = the activation energy of some reference reaction in the same reaction family
    alpha = the location of the transition state along the reaction coordinate (0 <= alpha <= 1)
    ΔH = the enthalpy of reaction

    Whereas ReactionRateCalculator uses the Eyring equation, here we are forced to use collision
    theory to estimate reaction rates.

    Args:
        reactants (list): list of MoleculeEntry objects
        products (list): list of MoleculeEntry objects
        ea_reference (float): activation energy reference point (in eV)
        delta_h_reference (float): reaction enthalpy reference point (in eV)
        reaction (dict, or None): optional. If None (default), the "reactants" and
        "products" lists will serve as the basis for a Reaction object which represents the
        balanced stoichiometric reaction. Otherwise, this dict will show the number of molecules
        present in the reaction for each reactant and each product in the reaction.
        alpha (float): the reaction coordinate (must between 0 and 1)
    """

    def __init__(self, reactants, products, ea_reference, delta_h_reference, alpha=0.5):

        self.ea_reference = ea_reference
        self.delta_h_reference = delta_h_reference
        self.alpha = alpha
        pro_mols = [p.mol_graph.molecule for p in products]
        rct_mols = [r.mol_graph.molecule for r in reactants]

        self.reactant_mass_factor = (
            product([r.composition.weight for r in rct_mols])
            / sum([r.composition.weight for r in rct_mols])
            * amu_to_kg
        )
        self.product_mass_factor = (
            product([p.composition.weight for p in pro_mols])
            / sum([p.composition.weight for p in pro_mols])
            * amu_to_kg
        )

        self.reactant_radius_factor = (
            pi
            * sum([(np.max(mol.distance_matrix) * (10 ** -10) / 2) for mol in rct_mols])
            ** 2
        )
        self.product_radius_factor = (
            pi
            * sum([(np.max(mol.distance_matrix) * (10 ** -10) / 2) for mol in pro_mols])
            ** 2
        )

        super().__init__(reactants, products, None)

    def calculate_act_energy(self, reverse=False):
        """
        Use the Bell-Evans-Polanyi principle to calculate the activation energy of the reaction.

        Args:
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            ea (float): the predicted energy of activation in eV
        """

        if reverse:
            enthalpy = -self.net_enthalpy
        else:
            enthalpy = self.net_enthalpy

        ea = self.ea_reference + self.alpha * (enthalpy - self.delta_h_reference)
        return ea

    def calculate_act_enthalpy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_enthalpy is not valid for " "BEPRateCalculator,"
        )

    def calculate_act_entropy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_entropy is not valid for " "BEPRateCalculator,"
        )

    def calculate_act_gibbs(self, temperature=ROOM_TEMP, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_gibbs is not valid for " "BEPRateCalculator,"
        )

    def calculate_activation_thermo(self, temperature=ROOM_TEMP, reverse=False):
        raise NotImplementedError(
            "Method calculate_activation_thermo is not valid for " "BEPRateCalculator,"
        )

    def calculate_rate_constant(self, temperature=ROOM_TEMP, reverse=False, kappa=None):
        """
        Calculate the rate constant predicted by collision theory.

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction
            kappa (None): not used for BEPRateCalculator

        Returns:
            k_rate (float): temperature-dependent rate constant
        """

        ea = self.calculate_act_energy(reverse=reverse)

        k_rate = np.exp(-ea / (KB * temperature))

        return k_rate

    def calculate_rate(
        self, concentrations, temperature=ROOM_TEMP, reverse=False, kappa=1.0
    ):
        """
        Calculate the rate using collision theory.

        Args:
            concentrations (list): concentrations of reactant molecules. Order of the reactants
                DOES matter.
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction
            kappa (float): here, kappa represents the steric factor (default 1.0, meaning that all
                collisions lead to appropriate conditions for a reaction

        Returns:
            rate (float): reaction rate, based on the stoichiometric rate law and the rate constant
        """

        k_rate = self.calculate_rate_constant(temperature=temperature, reverse=reverse)

        if reverse:
            exponents = self.rate_law["products"]
            mass_factor = self.product_mass_factor
            radius_factor = self.product_radius_factor
        else:
            exponents = self.rate_law["reactants"]
            mass_factor = self.reactant_mass_factor
            radius_factor = self.reactant_radius_factor

        # Radius factor will be 0 for single atoms
        if radius_factor == 0:
            radius_factor = 1

        total_exponent = sum(exponents)
        number_prefactor = (1000 * N_A) ** total_exponent
        concentration_factor = product(np.array(concentrations) ** exponents)
        root_factor = np.sqrt(8 * k * temperature / (pi * mass_factor))

        z = number_prefactor * concentration_factor * radius_factor * root_factor

        rate = z * kappa * k_rate
        return rate


class ExpandedBEPRateCalculator(ReactionRateCalculator):
    """
    A modified reaction rate calculator that uses a modified version of the Bell-Evans-Polanyi
    principle to predict the Gibbs free energy of activation (and, thus, the rate constants and
    reaction rates) of chemical reactions.

    The Bell-Evans-Polanyi principle states that, for reactions within a similar class or family,
    the difference in activation energy between the reactions is proportional to the difference in
    reaction enthalpy. That is,

    E_a = E_a,0 + alpha * ΔH_rel, where

    E_a = the activation energy of the reaction of interest
    E_a,0 = the activation energy of some reference reaction in the same reaction family
    alpha = the location of the transition state along the reaction coordinate (0 <= alpha <= 1)
    ΔH_rel = ΔH - ΔH_0 = the difference in enthalpy change between the reaction of interest and the
        reference reaction

    Here, we assume that

    ΔG_a = ΔG_a,0 + alpha * (ΔG), where

    ΔG_a = the Gibbs free energy of activation for the reaction of interest
    ΔG_a,0 = the Gibbs free energy of activation of some reference reaction in the same reaction
        family
    alpha = the location of he transition state along the reaction coordinate (0 <= alpha <= 1)
    ΔG_rel = ΔG - ΔG_0 = the difference in Gibbs free energy change between the reaction of interest
        and the reference reaction.

    Args:
        reactants (list): list of MoleculeEntry objects
        products (list): list of MoleculeEntry objects
        delta_ea_reference (float): activation energy reference point (in eV)
        delta_ha_reference (float): activation enthalpy reference point (in eV)
        delta_sa_reference (float): activation entropy reference point (in eV/K)
        delta_e_reference (float): reaction energy reference point (in eV)
        delta_h_reference (float): reaction enthalpy reference point (in eV)
        delta_s_reference (float): reaction entropy reference point (in eV/K)
        reaction (dict, or None): optional. If None (default), the "reactants" and
        "products" lists will serve as the basis for a Reaction object which represents the
        balanced stoichiometric reaction. Otherwise, this dict will show the number of molecules
        present in the reaction for each reactant and each product in the reaction.
        alpha (float): the reaction coordinate (must between 0 and 1)
    """

    def __init__(
        self,
        reactants,
        products,
        delta_ea_reference,
        delta_ha_reference,
        delta_sa_reference,
        delta_e_reference,
        delta_h_reference,
        delta_s_reference,
        alpha=0.5,
    ):

        # Reference values for activation properties
        self.delta_ea_reference = delta_ea_reference
        self.delta_ha_reference = delta_ha_reference
        self.delta_sa_reference = delta_sa_reference

        # Reference values for net reaction properties
        self.delta_e_reference = delta_e_reference
        self.delta_h_reference = delta_h_reference
        self.delta_s_reference = delta_s_reference

        # Reaction coordinate
        self.alpha = alpha

        super().__init__(reactants, products, None)

    def calculate_act_energy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_energy is not valid for " "ExpandedBEPRateCalculator,"
        )

    def calculate_act_enthalpy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_enthalpy is not valid for "
            "ExpandedBEPRateCalculator,"
        )

    def calculate_act_entropy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_entropy is not valid for " "ExpandedBEPCalculator,"
        )

    def calculate_act_gibbs(self, temperature=ROOM_TEMP, reverse=False):
        """
        Calculate Gibbs free energy of activation at a given temperature.

        ΔG = ΔH - T ΔS

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            delta_ga (float): Gibbs free energy of activation (in kcal/mol)
        """

        if reverse:
            delta_g = -self.calculate_net_gibbs(temperature)
        else:
            delta_g = self.calculate_net_gibbs(temperature)

        delta_g_ref = (
            self.delta_e_reference
            + self.delta_h_reference
            - temperature * self.delta_s_reference
        )
        delta_ga_ref = (
            self.delta_ea_reference
            + self.delta_ha_reference
            - temperature * self.delta_sa_reference
        )

        delta_ga = delta_ga_ref + self.alpha * (delta_g - delta_g_ref)

        return delta_ga

    def calculate_activation_thermo(self, temperature=ROOM_TEMP, reverse=False):
        raise NotImplementedError(
            "Method calculate_activation_thermo is not valid for "
            "ExpandedBEPRateCalculator,"
        )

    def calculate_rate_constant(self, temperature=ROOM_TEMP, reverse=False, kappa=1.0):
        """
        Calculate the rate constant k by the Eyring-Polanyi equation of transition state theory.

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction
            kappa (float): transmission coefficient (by default, we assume the assumptions of
                transition-state theory are valid, so kappa = 1.0

        Returns:
            k_rate (float): temperature-dependent rate constant
        """

        gibbs = self.calculate_act_gibbs(temperature=temperature, reverse=reverse)

        k_rate = kappa * KB * temperature / PLANCK * np.exp(-gibbs / (KB * temperature))
        return k_rate


class RedoxRateCalculator(ReactionRateCalculator):
    """
    This reaction rate calculator uses expressions from Marcus Theory to
    estimate the reaction rate for a reduction or oxidation reaction. It assumes
    that this reaction is between a single molecule in solution and an
    electrode, which is not treated explicitly. Future work may expand this
    class or develop another class to treat reduction and oxidation between
    two species in solution (a charge transfer reaction).

    The rate constant for a reduction or oxidation reaction is

    k = kappa * k_b * T / h * exp[-ΔG* / (k_b * T)]

    where kappa is the transmission coefficient (in this case, an electron
    tunnelling coefficient), Delta_G* is the energy barrier, k_b is the
    Boltzmann constant, T is the temperature in Kelvin, and h is the Planck
    constant.

    Here, we either assume that the reaction occurs adiabatically, in which
    case:

    kappa = 1

    or that the reaction is diabatic and has a simple exponential decay form:

    kappa = exp[-beta * R],

    where beta is some decay length (by default, 1.2 Angstrom^-1), and R is
    the distance to the electrode, in Angstrom.

    the energy barrier ΔG is based both on the reaction free energy

    ΔG = sum(G_product) - sum(G_reactant) - n*(electron free energy)

    where n is the number of electrons transferred (n positive for reduction,
    negative for oxidation), and the reorganization energy

    lambda = lambda_inner + lambda_outer,

    where lambda_inner is the inner reorganization energy, the energy required to
    repolarize the inner solvation shell after reduction/oxidation, and
    lambda_outer is the outer reorganization energy, the corresponding energy
    for the bulk solvent. Generally, the inner reorganization energy
    lambda_inner can be estimated directly, for instance by the four-point
    method (Nelsen, Blackstock, & Kim 1987). The outer reorganization energy,
    on the other hand, will be estimated as

    lambda_outer = (delta_e) ** 2 / (8 * pi * epsilon_0) * (1/r - 1/(2*R)) * (1/n^2 - 1/epsilon) (in J)
    lambda_outer = |delta_e| / (8 * pi * epsilon_0) * (1/r - 1/(2*R)) * (1/n^2 - 1/epsilon) (in eV)

    where delta_e is the change in fundamental charge (e = 1.602 * 10 **-19 C),
    epsilon_0 is tbe permittivity of the vacuum, r is the radius of the reactant
    including the inner solvation shell, n here is the index of refraction of
    the solvent, and epsilon is the dielectric constant of the solvent.

    Args:
        reactants (list): list of MoleculeEntry objects
        products (list): list of MoleculeEntry objects
        lambda_inner (float): Inner reorganization energy, in eV
        dielectric (float): Dielectric constant of the solvent (unitless)
        refractive (float): Refractive index of the solvent (unitless)
        electron_free_energy (float): Free energy of the electron in the
            electrode, in eV
        radius (float): Radius of the reactant/product, in Angstrom
        electrode_distance (float): Distance from the electrode surface, in
            Anstrom
    """

    def __init__(
        self,
        reactants,
        products,
        lambda_inner,
        dielectric,
        refractive,
        electron_free_energy,
        radius,
        electrode_distance,
        adiabatic=False,
        decay_constant=1.2,
    ):

        self.lambda_inner = lambda_inner
        self.dielectric = dielectric
        self.refractive = refractive
        self.electron_free_energy = electron_free_energy
        self.radius = radius
        self.electrode_distance = electrode_distance
        self.adiabatic = adiabatic
        self.decay_constant = decay_constant

        super().__init__(reactants, products, None)

        self.reactant_charge = sum([r.charge for r in reactants])
        self.product_charge = sum([p.charge for p in products])

    def update_calc(self, reference):
        """Update the rate calculator with a baseline reference values."""
        for kk in reference.keys():
            setattr(self, kk, reference.get(kk))

    def calculate_act_energy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_energy is not valid for " "RedoxRateCalculator,"
        )

    def calculate_act_enthalpy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_enthalpy is not valid for " "RedoxRateCalculator,"
        )

    def calculate_act_entropy(self, reverse=False):
        raise NotImplementedError(
            "Method calculate_act_entropy is not valid for " "RedoxRateCalculator,"
        )

    def calculate_outer_reorganization_energy(self):
        """
        Calculate the outer reorganization energy lambda_o using the Marcus
            method (Marcus 1965).

        Returns:
            lambda_outer (float), in eV
        """

        lambda_outer = abs(elementary_charge) / (8 * pi * epsilon_0)
        lambda_outer *= (1 / self.radius - 1 / (2 * self.electrode_distance)) * 10 ** 10
        lambda_outer *= 1 / self.refractive ** 2 - 1 / self.dielectric

        return lambda_outer

    def calculate_act_gibbs(self, temperature=ROOM_TEMP, reverse=False):
        """
        Calculate Gibbs free energy of activation at a given temperature.

        ΔG* = lambda/4 * (1 + ΔG/lambda)**2
        where lambda = lambda_inner + lambda_outer and

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction

        Returns:
            delta_ga (float): Gibbs free energy of activation (in kcal/mol)
        """

        lambda_total = self.lambda_inner + self.calculate_outer_reorganization_energy()

        charge_rct = self.reactant_charge
        charge_pro = self.product_charge

        if reverse:
            delta_g = -1 * self.calculate_net_gibbs(temperature=temperature)
            delta_g += self.electron_free_energy * (charge_rct - charge_pro)
        else:
            delta_g = self.calculate_net_gibbs(temperature=temperature)
            delta_g += self.electron_free_energy * (charge_pro - charge_rct)

        delta_ga = lambda_total / 4 * (1 + delta_g / lambda_total) ** 2

        return delta_ga

    def calculate_activation_thermo(self, temperature=ROOM_TEMP, reverse=False):
        raise NotImplementedError(
            "Method calculate_activation_thermo is not valid for "
            "RedoxRateCalculator,"
        )

    def calculate_rate_constant(self, temperature=ROOM_TEMP, reverse=False, kappa=1.0):
        """
        Calculate the rate constant k by the Eyring-Polanyi equation of transition state theory.

        Args:
            temperature (float): absolute temperature in Kelvin
            reverse (bool): if True (default False), consider the reverse reaction; otherwise,
                consider the forwards reaction
            kappa (float): transmission coefficient (not used in this function)

        Returns:
            k_rate (float): temperature-dependent rate constant
        """

        gibbs = self.calculate_act_gibbs(temperature=temperature, reverse=reverse)

        if not self.adiabatic:
            kappa = np.exp(-1 * self.decay_constant * self.electrode_distance)

        k_rate = kappa * KB * temperature / PLANCK * np.exp(-gibbs / (KB * temperature))

        return k_rate
