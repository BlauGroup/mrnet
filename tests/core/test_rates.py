# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import unittest
import os
import copy

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

# Not real molecules; just place-holders
# We're only interested in the math
mol_placeholder = Molecule(["H"], [[0.0, 0.0, 0.0]])


class ReactionRateCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        if ob:
            self.energies = [-271.553636516598, -78.5918513462683, -350.105998350078]
            self.enthalpies = [13.917, 34.596, 49.515]
            self.entropies = [67.357, 55.047, 84.265]

            self.rct_1 = MoleculeEntry(
                mol_placeholder,
                self.energies[0],
                enthalpy=self.enthalpies[0],
                entropy=self.entropies[0],
            )
            self.rct_2 = MoleculeEntry(
                mol_placeholder,
                self.energies[1],
                enthalpy=self.enthalpies[1],
                entropy=self.entropies[1],
            )
            self.pro = MoleculeEntry(
                mol_placeholder,
                self.energies[2],
                enthalpy=self.enthalpies[2],
                entropy=self.entropies[2],
            )

            self.ts = MoleculeEntry(
                mol_placeholder, -350.099875862606, enthalpy=48.560, entropy=83.607
            )
            self.reactants = [self.rct_1, self.rct_2]
            self.products = [self.pro]
            self.calc = ReactionRateCalculator(self.reactants, self.products, self.ts)

    def tearDown(self) -> None:
        if ob:
            del self.calc
            del self.ts
            del self.pro
            del self.rct_2
            del self.rct_1
            del self.entropies
            del self.enthalpies
            del self.energies

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_net_properties(self):
        self.assertAlmostEqual(
            self.calc.net_energy,
            (self.energies[2] - (self.energies[0] + self.energies[1])) * 27.2116,
            6,
        )
        self.assertEqual(
            self.calc.net_enthalpy,
            (self.enthalpies[2] - (self.enthalpies[0] + self.enthalpies[1]))
            * 0.0433641,
        )
        self.assertEqual(
            self.calc.net_entropy,
            (self.entropies[2] - (self.entropies[0] + self.entropies[1]))
            * 0.0000433641,
        )

        gibbs_300 = self.pro.get_free_energy(300) - (
            self.rct_1.get_free_energy(300) + self.rct_2.get_free_energy(300)
        )
        self.assertAlmostEqual(self.calc.calculate_net_gibbs(300), gibbs_300, 10)
        gibbs_100 = self.pro.get_free_energy(100) - (
            self.rct_1.get_free_energy(100) + self.rct_2.get_free_energy(100)
        )
        self.assertAlmostEqual(self.calc.calculate_net_gibbs(100.00), gibbs_100, 10)
        self.assertDictEqual(
            self.calc.calculate_net_thermo(),
            {
                "energy": self.calc.net_energy,
                "enthalpy": self.calc.net_enthalpy,
                "entropy": self.calc.net_entropy,
                "gibbs": self.calc.calculate_net_gibbs(),
            },
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_act_properties(self):
        trans_energy = self.ts.energy
        trans_enthalpy = self.ts.enthalpy
        trans_entropy = self.ts.entropy

        pro_energies = [p.energy for p in self.products]
        rct_energies = [r.energy for r in self.reactants]
        pro_enthalpies = [p.enthalpy for p in self.products]
        rct_enthalpies = [r.enthalpy for r in self.reactants]
        pro_entropies = [p.entropy for p in self.products]
        rct_entropies = [r.entropy for r in self.reactants]

        self.assertAlmostEqual(
            self.calc.calculate_act_energy(),
            (trans_energy - sum(rct_energies)) * 27.2116,
            6,
        )
        self.assertAlmostEqual(
            self.calc.calculate_act_energy(reverse=True),
            (trans_energy - sum(pro_energies)) * 27.2116,
            6,
        )

        self.assertEqual(
            self.calc.calculate_act_enthalpy(),
            (trans_enthalpy - sum(rct_enthalpies)) * 0.0433641,
        )
        self.assertEqual(
            self.calc.calculate_act_enthalpy(reverse=True),
            (trans_enthalpy - sum(pro_enthalpies)) * 0.0433641,
        )

        self.assertEqual(
            self.calc.calculate_act_entropy(),
            (trans_entropy - sum(rct_entropies)) * 0.0000433641,
        )
        self.assertEqual(
            self.calc.calculate_act_entropy(reverse=True),
            (trans_entropy - sum(pro_entropies)) * 0.0000433641,
        )

        gibbs_300 = self.calc.calculate_act_energy() + (
            self.calc.calculate_act_enthalpy() - 300 * self.calc.calculate_act_entropy()
        )
        gibbs_300_rev = self.calc.calculate_act_energy(reverse=True) + (
            self.calc.calculate_act_enthalpy(reverse=True)
            - 300 * self.calc.calculate_act_entropy(reverse=True)
        )
        gibbs_100 = (
            self.calc.calculate_act_energy()
            + self.calc.calculate_act_enthalpy()
            - 100 * self.calc.calculate_act_entropy()
        )
        self.assertEqual(self.calc.calculate_act_gibbs(300), gibbs_300)
        self.assertEqual(
            self.calc.calculate_act_gibbs(300, reverse=True), gibbs_300_rev
        )
        self.assertEqual(self.calc.calculate_act_gibbs(100), gibbs_100)

        self.assertEqual(
            self.calc.calculate_act_thermo(temperature=300.00),
            {
                "energy": self.calc.calculate_act_energy(),
                "enthalpy": self.calc.calculate_act_enthalpy(),
                "entropy": self.calc.calculate_act_entropy(),
                "gibbs": self.calc.calculate_act_gibbs(300),
            },
        )
        self.assertEqual(
            self.calc.calculate_act_thermo(temperature=300.00, reverse=True),
            {
                "energy": self.calc.calculate_act_energy(reverse=True),
                "enthalpy": self.calc.calculate_act_enthalpy(reverse=True),
                "entropy": self.calc.calculate_act_entropy(reverse=True),
                "gibbs": self.calc.calculate_act_gibbs(300, reverse=True),
            },
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_rate_constant(self):

        gibbs_300 = self.calc.calculate_act_gibbs(300)
        gibbs_300_rev = self.calc.calculate_act_gibbs(300, reverse=True)
        gibbs_600 = self.calc.calculate_act_gibbs(600)

        # Test normal forwards and reverse behavior
        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=300.0),
            k * 300 / h * np.exp(-gibbs_300 / (8.617333262 * 10 ** -5 * 300)),
        )
        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=600),
            k * 600 / h * np.exp(-gibbs_600 / (8.617333262 * 10 ** -5 * 600)),
        )
        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=300.0, reverse=True),
            k * 300 / h * np.exp(-gibbs_300_rev / (8.617333262 * 10 ** -5 * 300)),
        )

        # Test effect of kappa
        self.assertEqual(
            self.calc.calculate_rate_constant(),
            self.calc.calculate_rate_constant(kappa=0.5) * 2,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_rates(self):

        rate_constant = self.calc.calculate_rate_constant()
        rate_constant_600 = self.calc.calculate_rate_constant(temperature=600)
        rate_constant_rev = self.calc.calculate_rate_constant(reverse=True)
        base_rate = rate_constant

        self.assertAlmostEqual(self.calc.calculate_rate([1, 1]), base_rate)
        self.assertAlmostEqual(self.calc.calculate_rate([1, 0.5]), base_rate / 2, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([0.5, 1]), base_rate / 2, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([0.5, 0.5]), base_rate / 4, 8)
        self.assertAlmostEqual(
            self.calc.calculate_rate([1], reverse=True), rate_constant_rev, 8
        )
        self.assertAlmostEqual(
            self.calc.calculate_rate([1, 1], temperature=600), rate_constant_600, 8
        )


class BEPReactionRateCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:

        if ob:
            self.energies = [-271.553636516598, -78.5918513462683, -350.105998350078]
            self.enthalpies = [13.917, 34.596, 49.515]
            self.entropies = [67.357, 55.047, 84.265]

            self.rct_1 = MoleculeEntry(
                mol_placeholder,
                self.energies[0],
                enthalpy=self.enthalpies[0],
                entropy=self.entropies[0],
            )
            self.rct_2 = MoleculeEntry(
                mol_placeholder,
                self.energies[1],
                enthalpy=self.enthalpies[1],
                entropy=self.entropies[1],
            )
            self.pro = MoleculeEntry(
                mol_placeholder,
                self.energies[2],
                enthalpy=self.enthalpies[2],
                entropy=self.entropies[2],
            )

            self.calc = BEPRateCalculator(
                [self.rct_1, self.rct_2], [self.pro], 1.718386088799889, 1.722
            )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_act_properties(self):
        self.assertAlmostEqual(
            self.calc.calculate_act_energy(),
            self.calc.ea_reference
            + 0.5 * (self.calc.net_enthalpy - self.calc.delta_h_reference),
            6,
        )
        self.assertAlmostEqual(
            self.calc.calculate_act_energy(reverse=True),
            self.calc.ea_reference
            + 0.5 * (-1 * self.calc.net_enthalpy - self.calc.delta_h_reference),
            6,
        )

        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_enthalpy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_entropy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_gibbs(300)
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_thermo(temperature=300.00)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_rate_constant(self):
        rate_constant = np.exp(
            -self.calc.calculate_act_energy() / (8.617333262 * 10 ** -5 * 300)
        )
        rate_constant_600 = np.exp(
            -self.calc.calculate_act_energy() / (8.617333262 * 10 ** -5 * 600)
        )

        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=300), rate_constant
        )
        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=600), rate_constant_600
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_rates(self):
        base_rate = self.calc.calculate_rate([1, 1])
        rate_600 = self.calc.calculate_rate([1, 1], temperature=600)

        self.assertAlmostEqual(self.calc.calculate_rate([1, 1]) / base_rate, 1, 6)
        self.assertAlmostEqual(
            self.calc.calculate_rate([1, 0.5]) / (base_rate / 2), 1, 6
        )
        self.assertAlmostEqual(
            self.calc.calculate_rate([0.5, 1]) / (base_rate / 2), 1, 6
        )
        self.assertAlmostEqual(
            self.calc.calculate_rate([0.5, 0.5]) / (base_rate / 4), 1, 6
        )
        self.assertAlmostEqual(
            self.calc.calculate_rate([1, 1], kappa=0.5) / (base_rate / 2), 1, 6
        )

        self.assertAlmostEqual(
            self.calc.calculate_rate([1, 1], temperature=600) / rate_600, 1, 6
        )


class ExpandedBEPReactionRateCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:

        if ob:
            self.energies = [-271.553636516598, -78.5918513462683, -350.105998350078]
            self.enthalpies = [13.917, 34.596, 49.515]
            self.entropies = [67.357, 55.047, 84.265]

            self.rct_1 = MoleculeEntry(
                mol_placeholder,
                self.energies[0],
                enthalpy=self.enthalpies[0],
                entropy=self.entropies[0],
            )
            self.rct_2 = MoleculeEntry(
                mol_placeholder,
                self.energies[1],
                enthalpy=self.enthalpies[1],
                entropy=self.entropies[1],
            )
            self.pro = MoleculeEntry(
                mol_placeholder,
                self.energies[2],
                enthalpy=self.enthalpies[2],
                entropy=self.entropies[2],
            )

            self.calc = ExpandedBEPRateCalculator(
                [self.rct_1, self.rct_2], [self.pro], 1.71, 0.1, -0.05, 1.8, 0.1, 0.05
            )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_act_properties(self):

        delta_g_ref = (
            self.calc.delta_e_reference
            + self.calc.delta_h_reference
            - 300 * self.calc.delta_s_reference
        )
        delta_g = self.calc.calculate_net_gibbs(300)
        delta_g_rev = -delta_g

        delta_g_ref_600 = (
            self.calc.delta_e_reference
            + self.calc.delta_h_reference
            - 600 * self.calc.delta_s_reference
        )
        delta_g_600 = self.calc.calculate_net_gibbs(600)

        delta_ga_ref_300 = self.calc.delta_ea_reference + (
            self.calc.delta_ha_reference - 300 * self.calc.delta_sa_reference
        )
        delta_ga_ref_600 = self.calc.delta_ea_reference + (
            self.calc.delta_ha_reference - 600 * self.calc.delta_sa_reference
        )

        self.assertAlmostEqual(
            self.calc.calculate_act_gibbs(300),
            delta_ga_ref_300 + self.calc.alpha * (delta_g - delta_g_ref),
        )
        self.assertAlmostEqual(
            self.calc.calculate_act_gibbs(300, reverse=True),
            delta_ga_ref_300 + self.calc.alpha * (delta_g_rev - delta_g_ref),
        )
        self.assertAlmostEqual(
            self.calc.calculate_act_gibbs(600),
            delta_ga_ref_600 + self.calc.alpha * (delta_g_600 - delta_g_ref_600),
        )

        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_energy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_enthalpy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_entropy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_thermo(temperature=300.00)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_rate_constant(self):
        gibbs_300 = self.calc.calculate_act_gibbs(300)
        gibbs_600 = self.calc.calculate_act_gibbs(600)

        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=300),
            k * 300 / h * np.exp(-gibbs_300 / (8.617333262 * 10 ** -5 * 300)),
        )
        self.assertEqual(
            self.calc.calculate_rate_constant(temperature=600),
            k * 600 / h * np.exp(-gibbs_600 / (8.617333262 * 10 ** -5 * 600)),
        )

        # Test effect of kappa
        self.assertEqual(
            self.calc.calculate_rate_constant(),
            self.calc.calculate_rate_constant(kappa=0.5) * 2,
        )


class RedoxRateCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:

        if ob:
            self.energies = [-349.88738062842, -349.955817900195]
            self.enthalpies = [53.623, 51.853]
            self.entropies = [82.846, 79.595]

            rct_mol = copy.deepcopy(mol_placeholder)
            rct_mol.set_charge_and_spin(charge=1)

            pro_mol = copy.deepcopy(mol_placeholder)
            pro_mol.set_charge_and_spin(charge=0)

            self.rct = MoleculeEntry(
                rct_mol,
                self.energies[0],
                enthalpy=self.enthalpies[0],
                entropy=self.entropies[0],
            )
            self.pro = MoleculeEntry(
                pro_mol,
                self.energies[1],
                enthalpy=self.enthalpies[1],
                entropy=self.entropies[1],
            )

            self.calc = RedoxRateCalculator(
                [self.rct], [self.pro], 1.031373321805404, 18.5, 1.415, -1.897, 7.5, 5
            )

            self.calc_adiabatic = RedoxRateCalculator(
                [self.rct],
                [self.pro],
                1.031373321805404,
                18.5,
                1.415,
                -1.897,
                7.5,
                5,
                adiabatic=True,
            )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_act_properties(self):
        self.assertAlmostEqual(
            self.calc.calculate_act_gibbs(temperature=300), 0.284698735, 9
        )
        self.assertAlmostEqual(
            self.calc.calculate_act_gibbs(temperature=300, reverse=True), 0.284433478, 9
        )
        self.assertAlmostEqual(
            self.calc.calculate_act_gibbs(temperature=600), 0.306243023, 9
        )

        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_energy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_enthalpy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_entropy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_thermo(temperature=300.00)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_rate_constant(self):
        self.assertAlmostEqual(
            self.calc.calculate_rate_constant(temperature=300) / 255536.74880926133,
            1.0,
            4,
        )
        self.assertAlmostEqual(
            self.calc.calculate_rate_constant(temperature=300, reverse=True)
            / 258172.2056825794,
            1.0,
            4,
        )
        self.assertAlmostEqual(
            self.calc.calculate_rate_constant(temperature=600) / 82962806.19389883,
            1.0,
            4,
        )

        self.assertAlmostEqual(
            self.calc_adiabatic.calculate_rate_constant() / 95631480.11437328, 1.0, 4
        )


if __name__ == "__main__":
    unittest.main()
