# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import copy

import pytest

import numpy as np
from scipy.constants import h, k

from pymatgen.core.structure import Molecule

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.rates import (
    ReactionRateCalculator,
    BEPRateCalculator,
    ExpandedBEPRateCalculator,
    RedoxRateCalculator,
)

try:
    import openbabel as ob
except ImportError:
    ob = None

__author__ = "Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "September 2019"

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

mol_placeholder = Molecule(["H"], [[0.0, 0.0, 0.0]])

energies = [-271.553636516598, -78.5918513462683, -350.105998350078]
enthalpies = [13.917, 34.596, 49.515]
entropies = [67.357, 55.047, 84.265]

rct_1 = MoleculeEntry(
    mol_placeholder, energies[0], enthalpy=enthalpies[0], entropy=entropies[0],
)
rct_2 = MoleculeEntry(
    mol_placeholder, energies[1], enthalpy=enthalpies[1], entropy=entropies[1],
)
pro = MoleculeEntry(
    mol_placeholder, energies[2], enthalpy=enthalpies[2], entropy=entropies[2],
)

ts = MoleculeEntry(mol_placeholder, -350.099875862606, enthalpy=48.560, entropy=83.607)


def make_reaction_rate_calc():
    calc = ReactionRateCalculator([rct_1, rct_2], [pro], ts)
    return calc


def make_bep_rate_calc():
    calc = BEPRateCalculator([rct_1, rct_2], [pro], 1.718386088799889, 1.722)
    return calc


def make_exp_bep_rate_calc():
    calc = ExpandedBEPRateCalculator(
        [rct_1, rct_2], [pro], 1.71, 0.1, -0.05, 1.8, 0.1, 0.05
    )
    return calc


def make_redox_calc():
    energies = [-349.88738062842, -349.955817900195]
    enthalpies = [53.623, 51.853]
    entropies = [82.846, 79.595]

    rct_mol = copy.deepcopy(mol_placeholder)
    rct_mol.set_charge_and_spin(charge=1)

    pro_mol = copy.deepcopy(mol_placeholder)
    pro_mol.set_charge_and_spin(charge=0)

    rct = MoleculeEntry(
        rct_mol, energies[0], enthalpy=enthalpies[0], entropy=entropies[0],
    )
    pro = MoleculeEntry(
        pro_mol, energies[1], enthalpy=enthalpies[1], entropy=entropies[1],
    )

    calc = RedoxRateCalculator(
        [rct], [pro], 1.031373321805404, 18.5, 1.415, -1.897, 7.5, 5
    )
    return calc


class TestReactionRateCalculator:
    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_net_properties():
        calc = make_reaction_rate_calc()

        assert (
            round(
                calc.net_energy - (energies[2] - (energies[0] + energies[1])) * 27.2116,
                6,
            )
            == 0
        )

        assert (
            calc.net_enthalpy
            == (enthalpies[2] - (enthalpies[0] + enthalpies[1])) * 0.0433641
        )

        assert (
            calc.net_entropy
            == (entropies[2] - (entropies[0] + entropies[1])) * 0.0000433641
        )

        gibbs_300 = pro.get_free_energy(300) - (
            rct_1.get_free_energy(300) + rct_2.get_free_energy(300)
        )

        assert calc.calculate_net_gibbs(300) == gibbs_300

        gibbs_100 = pro.get_free_energy(100) - (
            rct_1.get_free_energy(100) + rct_2.get_free_energy(100)
        )

        assert calc.calculate_net_gibbs(100) == gibbs_100

        thermo_dict = {
            "energy": calc.net_energy,
            "enthalpy": calc.net_enthalpy,
            "entropy": calc.net_entropy,
            "gibbs": calc.calculate_net_gibbs(),
        }

        assert calc.calculate_net_thermo() == thermo_dict

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_act_properties():
        calc = make_reaction_rate_calc()

        trans_energy = ts.energy
        trans_enthalpy = ts.enthalpy
        trans_entropy = ts.entropy

        pro_energies = [p.energy for p in calc.products]
        rct_energies = [r.energy for r in calc.reactants]
        pro_enthalpies = [p.enthalpy for p in calc.products]
        rct_enthalpies = [r.enthalpy for r in calc.reactants]
        pro_entropies = [p.entropy for p in calc.products]
        rct_entropies = [r.entropy for r in calc.reactants]

        assert (
            round(
                calc.calculate_act_energy()
                - (trans_energy - sum(rct_energies)) * 27.2116,
                6,
            )
            == 0
        )
        assert (
            round(
                calc.calculate_act_energy(reverse=True)
                - (trans_energy - sum(pro_energies)) * 27.2116,
                6,
            )
            == 0
        )

        assert (
            calc.calculate_act_enthalpy()
            == (trans_enthalpy - sum(rct_enthalpies)) * 0.0433641
        )
        assert (
            calc.calculate_act_enthalpy(reverse=True)
            == (trans_enthalpy - sum(pro_enthalpies)) * 0.0433641
        )

        assert (
            calc.calculate_act_entropy()
            == (trans_entropy - sum(rct_entropies)) * 0.0000433641
        )
        assert (
            calc.calculate_act_entropy(reverse=True)
            == (trans_entropy - sum(pro_entropies)) * 0.0000433641
        )

        gibbs_300 = calc.calculate_act_energy() + (
            calc.calculate_act_enthalpy() - 300 * calc.calculate_act_entropy()
        )
        gibbs_300_rev = calc.calculate_act_energy(reverse=True) + (
            calc.calculate_act_enthalpy(reverse=True)
            - 300 * calc.calculate_act_entropy(reverse=True)
        )
        gibbs_100 = (
            calc.calculate_act_energy()
            + calc.calculate_act_enthalpy()
            - 100 * calc.calculate_act_entropy()
        )

        assert calc.calculate_act_gibbs(300) == gibbs_300
        assert calc.calculate_act_gibbs(300, reverse=True) == gibbs_300_rev
        assert calc.calculate_act_gibbs(100) == gibbs_100

        thermo_dict = {
            "energy": calc.calculate_act_energy(),
            "enthalpy": calc.calculate_act_enthalpy(),
            "entropy": calc.calculate_act_entropy(),
            "gibbs": calc.calculate_act_gibbs(300),
        }

        thermo_dict_reversed = {
            "energy": calc.calculate_act_energy(reverse=True),
            "enthalpy": calc.calculate_act_enthalpy(reverse=True),
            "entropy": calc.calculate_act_entropy(reverse=True),
            "gibbs": calc.calculate_act_gibbs(300, reverse=True),
        }

        assert calc.calculate_act_thermo(temperature=300.00) == thermo_dict
        assert (
            calc.calculate_act_thermo(temperature=300.00, reverse=True)
            == thermo_dict_reversed
        )

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_rate_constant():
        calc = make_reaction_rate_calc()

        gibbs_300 = calc.calculate_act_gibbs(300)
        gibbs_300_rev = calc.calculate_act_gibbs(300, reverse=True)
        gibbs_600 = calc.calculate_act_gibbs(600)

        # Test normal forwards and reverse behavior
        k_for = k * 300 / h * np.exp(-gibbs_300 / (8.617333262 * 10 ** -5 * 300))
        k_600_for = k * 600 / h * np.exp(-gibbs_600 / (8.617333262 * 10 ** -5 * 600))
        k_rev = k * 300 / h * np.exp(-gibbs_300_rev / (8.617333262 * 10 ** -5 * 300))

        assert calc.calculate_rate_constant(temperature=300.00) == k_for
        assert calc.calculate_rate_constant(temperature=600.00) == k_600_for
        assert calc.calculate_rate_constant(temperature=300.00, reverse=True) == k_rev

        # Test effect of kappa
        assert (
            calc.calculate_rate_constant()
            == calc.calculate_rate_constant(kappa=0.5) * 2
        )

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_rates():
        calc = make_reaction_rate_calc()

        rate_constant = calc.calculate_rate_constant()
        rate_constant_600 = calc.calculate_rate_constant(temperature=600)
        rate_constant_rev = calc.calculate_rate_constant(reverse=True)
        base_rate = rate_constant

        assert round(calc.calculate_rate([1, 1]) - base_rate) == 0
        assert round(calc.calculate_rate([1, 0.5]) - base_rate / 2) == 0
        assert round(calc.calculate_rate([0.5, 1]) - base_rate / 2) == 0
        assert round(calc.calculate_rate([0.5, 0.5]) - base_rate / 4) == 0
        assert round(calc.calculate_rate([1], reverse=True) - rate_constant_rev) == 0
        assert (
            round(calc.calculate_rate([1, 1], temperature=600.00) - rate_constant_600)
            == 0
        )


class TestBEPReactionRateCalculator:
    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_act_properties():
        calc = make_bep_rate_calc()

        assert (
            round(
                calc.calculate_act_energy()
                - (
                    calc.ea_reference
                    + 0.5 * (calc.net_enthalpy - calc.delta_h_reference)
                )
            )
            == 0
        )
        assert (
            round(
                calc.calculate_act_energy(reverse=True)
                - (
                    calc.ea_reference
                    + 0.5 * (-1 * calc.net_enthalpy - calc.delta_h_reference)
                )
            )
            == 0
        )

        with pytest.raises(NotImplementedError):
            calc.calculate_act_enthalpy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_entropy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_gibbs(300)
        with pytest.raises(NotImplementedError):
            calc.calculate_act_thermo(temperature=300.00)

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_rate_constant():
        calc = make_bep_rate_calc()

        rate_constant = np.exp(
            -calc.calculate_act_energy() / (8.617333262 * 10 ** -5 * 300)
        )
        rate_constant_600 = np.exp(
            -calc.calculate_act_energy() / (8.617333262 * 10 ** -5 * 600)
        )

        assert calc.calculate_rate_constant(temperature=300) == rate_constant
        assert calc.calculate_rate_constant(temperature=600) == rate_constant_600

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_rates():
        calc = make_bep_rate_calc()

        base_rate = calc.calculate_rate([1, 1])
        rate_600 = calc.calculate_rate([1, 1], temperature=600)

        assert round(calc.calculate_rate([1, 1]) - base_rate) == 0
        assert round(calc.calculate_rate([1, 0.5]) / (base_rate / 2) - 1) == 0
        assert round(calc.calculate_rate([0.5, 1.0]) / (base_rate / 2) - 1) == 0
        assert round(calc.calculate_rate([0.5, 0.5]) / (base_rate / 4) - 1) == 0
        assert round(calc.calculate_rate([1, 1], kappa=0.5) / (base_rate / 2) - 1) == 0
        assert round(calc.calculate_rate([1, 1], temperature=600) / rate_600 - 1) == 0


class TestExpandedBEPReactionRateCalculator:
    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_act_properties():
        calc = make_exp_bep_rate_calc()

        delta_g_ref = (
            calc.delta_e_reference
            + calc.delta_h_reference
            - 300 * calc.delta_s_reference
        )
        delta_g = calc.calculate_net_gibbs(300)
        delta_g_rev = -delta_g

        delta_g_ref_600 = (
            calc.delta_e_reference
            + calc.delta_h_reference
            - 600 * calc.delta_s_reference
        )
        delta_g_600 = calc.calculate_net_gibbs(600)

        delta_ga_ref_300 = calc.delta_ea_reference + (
            calc.delta_ha_reference - 300 * calc.delta_sa_reference
        )
        delta_ga_ref_600 = calc.delta_ea_reference + (
            calc.delta_ha_reference - 600 * calc.delta_sa_reference
        )

        assert (
            round(
                calc.calculate_act_gibbs(300)
                - (delta_ga_ref_300 + calc.alpha * (delta_g - delta_g_ref))
            )
            == 0
        )
        assert (
            round(
                calc.calculate_act_gibbs(300, reverse=True)
                - (delta_ga_ref_300 + calc.alpha * (delta_g_rev - delta_g_ref))
            )
            == 0
        )
        assert (
            round(
                calc.calculate_act_gibbs(600)
                - (delta_ga_ref_600 + calc.alpha * (delta_g_600 - delta_g_ref_600))
            )
            == 0
        )

        with pytest.raises(NotImplementedError):
            calc.calculate_act_energy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_enthalpy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_entropy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_thermo(temperature=300.00)

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_rate_constant():
        calc = make_exp_bep_rate_calc()
        gibbs_300 = calc.calculate_act_gibbs(300)
        gibbs_600 = calc.calculate_act_gibbs(600)

        assert calc.calculate_rate_constant(temperature=300) == k * 300 / h * np.exp(
            -gibbs_300 / (8.617333262 * 10 ** -5 * 300)
        )
        assert (
            calc.calculate_rate_constant(temperature=600)
            == k * 600 / h * np.exp(-gibbs_600 / (8.617333262 * 10 ** -5 * 600)),
        )

        # Test effect of kappa
        assert (
            calc.calculate_rate_constant()
            == calc.calculate_rate_constant(kappa=0.5) * 2
        )


class TestRedoxRateCalculator:
    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_act_properties():
        calc = make_redox_calc()

        assert round(calc.calculate_act_gibbs(temperature=300) - 0.284698735) == 0
        assert (
            round(calc.calculate_act_gibbs(temperature=300, reverse=True) - 0.284433478)
            == 0
        )
        assert round(calc.calculate_act_gibbs(temperature=600) - 0.306243023) == 0

        with pytest.raises(NotImplementedError):
            calc.calculate_act_energy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_enthalpy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_entropy()
        with pytest.raises(NotImplementedError):
            calc.calculate_act_thermo(temperature=300.00)

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_rate_constant():
        calc = make_redox_calc()
        assert (
            round(calc.calculate_rate_constant(temperature=300) - 255536.74880926133, 4)
            == 0
        )
        assert (
            round(
                calc.calculate_rate_constant(temperature=300, reverse=True)
                - 258172.2056825794,
                4,
            )
            == 0
        )
        assert (
            round(calc.calculate_rate_constant(temperature=600) - 82962806.19389883, 4)
            == 0
        )
