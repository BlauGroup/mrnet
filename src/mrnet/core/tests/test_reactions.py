# coding: utf-8
import os
import unittest

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender

from mrnet.core.reactions import (
    RedoxReaction,
    IntramolSingleBondChangeReaction,
    IntermolecularReaction,
    CoordinationBondChangeReaction,
)
from mrnet.core.reactions import bucket_mol_entries, unbucket_mol_entries
from mrnet.network.reaction_network import ReactionNetwork

from monty.serialization import loadfn

try:
    import openbabel as ob
except ImportError:
    ob = None


test_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "test_files", "reaction_network_files",
)


class TestRedoxReaction(PymatgenTest):
    @classmethod
    def setUpClass(cls) -> None:
        if ob:
            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir, "LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(
                    molecule=mol, energy=E, enthalpy=H, entropy=S, entry_id=entry["task_id"],
                )
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
            )
            cls.EC_mg = metal_edge_extender(EC_mg)
            cls.EC_0_entry = None
            cls.EC_minus_entry = None
            cls.EC_1_entry = None

            for entry in cls.LiEC_reextended_entries:
                if (
                    entry.formula == "C3 H4 O3"
                    and entry.charge == 0
                    and entry.Nbonds == 10
                    and cls.EC_mg.isomorphic_to(entry.mol_graph)
                ):
                    cls.EC_0_entry = entry
                elif (
                    entry.formula == "C3 H4 O3"
                    and entry.charge == -1
                    and entry.Nbonds == 10
                    and cls.EC_mg.isomorphic_to(entry.mol_graph)
                ):
                    cls.EC_minus_entry = entry
                elif (
                    entry.formula == "C3 H4 O3"
                    and entry.charge == 1
                    and entry.Nbonds == 10
                    and cls.EC_mg.isomorphic_to(entry.mol_graph)
                ):
                    cls.EC_1_entry = entry
                if (
                    cls.EC_0_entry is not None
                    and cls.EC_minus_entry is not None
                    and cls.EC_1_entry is not None
                ):
                    break

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)

        EC_0_ind = None
        EC_1_ind = None
        EC_0_RN_entry = None
        EC_1_RN_entry = None
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_0_ind = entry.parameters["ind"]
                EC_0_RN_entry = entry
                break
        for entry in RN.entries["C3 H4 O3"][10][1]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_1_ind = entry.parameters["ind"]
                EC_1_RN_entry = entry
                break

        reaction = RedoxReaction(EC_0_RN_entry, EC_1_RN_entry)
        reaction.electron_free_energy = -2.15
        graph = reaction.graph_representation()

        self.assertCountEqual(
            list(graph.nodes),
            [
                EC_0_ind,
                EC_1_ind,
                str(EC_0_ind) + "," + str(EC_1_ind),
                str(EC_1_ind) + "," + str(EC_0_ind),
            ],
        )
        self.assertEqual(len(graph.edges), 4)
        self.assertEqual(
            graph.get_edge_data(EC_0_ind, str(EC_0_ind) + "," + str(EC_1_ind))["softplus"],
            5.629805462349386,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = RedoxReaction.generate(RN.entries)

        self.assertEqual(len(reactions), 273)

        # for r in reactions:
        #    if r.reactant == self.EC_0_entry:
        #        self.assertEqual(r.product.entry_id, self.EC_1_entry.entry_id)
        #    if r.reactant == self.EC_minus_entry:
        #        self.assertEqual(r.product.entry_id, self.EC_0_entry.entry_id)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        entries = bucket_mol_entries([self.EC_minus_entry, self.EC_0_entry, self.EC_1_entry])

        reactions, families = RedoxReaction.generate(entries)
        self.assertEqual(len(reactions), 2)

        for r in reactions:
            self.assertEqual(
                r.reactants_atom_mapping,
                [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}],
            )
            self.assertEqual(
                r.products_atom_mapping,
                [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}],
            )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = RedoxReaction(self.EC_0_entry, self.EC_1_entry)
        reaction.electron_free_energy = -2.15
        reaction.free_energy()
        self.assertEqual(reaction.free_energy_A, 6.231346035181195)
        self.assertEqual(reaction.free_energy_B, -6.231346035181195)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = RedoxReaction(self.EC_0_entry, self.EC_1_entry)
        reaction.electron_free_energy = -2.15
        reaction.energy()
        self.assertEqual(reaction.energy_A, 0.3149076465170424)
        self.assertEqual(reaction.energy_B, -0.3149076465170424)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):
        reaction = RedoxReaction(self.EC_0_entry, self.EC_1_entry)
        self.assertEqual(reaction.rxn_type_A, "One electron oxidation")
        self.assertEqual(reaction.rxn_type_B, "One electron reduction")


class TestIntramolSingleBondChangeReaction(PymatgenTest):
    @classmethod
    def setUpClass(cls) -> None:
        if ob:
            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir, "LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(
                    molecule=mol, energy=E, enthalpy=H, entropy=S, entry_id=entry["task_id"],
                )
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            LiEC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
            )
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            LiEC_RO_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC_RO.xyz")), OpenBabelNN()
            )
            cls.LiEC_RO_mg = metal_edge_extender(LiEC_RO_mg)

            cls.LiEC_entry = None
            cls.LiEC_RO_entry = None

            for entry in cls.LiEC_reextended_entries:
                if (
                    entry.formula == "C3 H4 Li1 O3"
                    and entry.charge == 0
                    and entry.Nbonds == 12
                    and cls.LiEC_mg.isomorphic_to(entry.mol_graph)
                ):
                    cls.LiEC_entry = entry
                elif (
                    entry.formula == "C3 H4 Li1 O3"
                    and entry.charge == 0
                    and entry.Nbonds == 11
                    and cls.LiEC_RO_mg.isomorphic_to(entry.mol_graph)
                ):
                    cls.LiEC_RO_entry = entry
                if cls.LiEC_entry is not None and cls.LiEC_RO_entry is not None:
                    break

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        # print(RN.entries["C3 H4 Li1 O3"][11][0][0].molecule)

        LiEC_ind = None
        LiEC_RO_ind = None
        LiEC_RN_entry = None
        LiEC_RO_RN_entry = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][0]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                LiEC_RN_entry = entry
                break
        for entry in RN.entries["C3 H4 Li1 O3"][11][0]:
            if self.LiEC_RO_mg.isomorphic_to(entry.mol_graph):
                LiEC_RO_ind = entry.parameters["ind"]
                LiEC_RO_RN_entry = entry
                break
        reaction = IntramolSingleBondChangeReaction(LiEC_RN_entry, LiEC_RO_RN_entry)
        reaction.electron_free_energy = -2.15
        graph = reaction.graph_representation()
        print(graph.nodes, graph.edges)
        print(graph.get_edge_data(LiEC_ind, str(LiEC_ind) + "," + str(LiEC_RO_ind)))
        self.assertCountEqual(
            list(graph.nodes),
            [
                LiEC_ind,
                LiEC_RO_ind,
                str(LiEC_ind) + "," + str(LiEC_RO_ind),
                str(LiEC_RO_ind) + "," + str(LiEC_ind),
            ],
        )
        self.assertEqual(len(graph.edges), 4)
        self.assertEqual(
            graph.get_edge_data(LiEC_ind, str(LiEC_ind) + "," + str(LiEC_RO_ind))["softplus"],
            0.15092362164364986,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = IntramolSingleBondChangeReaction.generate(RN.entries)
        self.assertEqual(len(reactions), 93)

        for r in reactions:
            # TODO (mjwen) this is never run for two reasons:
            #  1. It should be:
            #  if r.reactant == self.LiEC_RO_entry:
            #     self.assertEqual(r.product.entry_id, self.LiEC_entry.entry_id)
            #  since this class generates bond formation reactions
            #  Even, after fixing 1, there is still another problem.
            #  2. There are multiple MoleculeEntry with the same `formula`,`Nbonds`,
            #  and `charge` as self.LiEC_entry. In `setUpClass`, one of such
            #  MoleculeEntry is set to self.LiEC_entry,
            #  but in IntramolSingleBondChangeReaction, another MoleculeEntry will be
            #  used as the reactant. This happens because in both `setUpClass` and
            #  `IntramolSingleBondChangeReaction`, the code `break` when one entry is
            #  found.
            #  To fix this, we can either clean the input data to make sure there is
            #  only one LiEC, or we do some bookkeeping in `setUpClass` and then make
            #  the correct check.

            # if r.reactant == self.LiEC_entry:
            #    self.assertEqual(r.product.entry_id, self.LiEC_RO_entry.entry_id)
            pass

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        entries = bucket_mol_entries([self.LiEC_RO_entry, self.LiEC_entry])

        reactions, families = IntramolSingleBondChangeReaction.generate(entries)
        self.assertEqual(len(reactions), 1)
        rxn = reactions[0]
        self.assertEqual(rxn.reactant_ids[0], self.LiEC_RO_entry.entry_id)
        self.assertEqual(rxn.product_ids[0], self.LiEC_entry.entry_id)

        self.assertEqual(
            rxn.reactants_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )
        self.assertEqual(
            rxn.products_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = IntramolSingleBondChangeReaction(self.LiEC_entry, self.LiEC_RO_entry)
        reaction.free_energy()
        self.assertEqual(reaction.free_energy_A, -1.1988634269218892)
        self.assertEqual(reaction.free_energy_B, 1.1988634269218892)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = IntramolSingleBondChangeReaction(self.LiEC_entry, self.LiEC_RO_entry)
        reaction.energy()
        self.assertEqual(reaction.energy_A, -0.03746218086303088)
        self.assertEqual(reaction.energy_B, 0.03746218086303088)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = IntramolSingleBondChangeReaction(self.LiEC_entry, self.LiEC_RO_entry)
        self.assertEqual(reaction.rxn_type_A, "Intramolecular single bond formation")
        self.assertEqual(reaction.rxn_type_B, "Intramolecular single bond breakage")


class TestIntermolecularReaction(PymatgenTest):
    @classmethod
    def setUpClass(cls) -> None:
        if ob:
            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir, "LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(
                    molecule=mol, energy=E, enthalpy=H, entropy=S, entry_id=entry["task_id"],
                )
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            C2H4_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "C2H4.xyz")), OpenBabelNN()
            )
            cls.C2H4_mg = metal_edge_extender(C2H4_mg)

            LiEC_RO_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC_RO.xyz")), OpenBabelNN()
            )
            cls.LiEC_RO_mg = metal_edge_extender(LiEC_RO_mg)

            C1Li1O3_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "C1Li1O3.xyz")), OpenBabelNN()
            )
            cls.C1Li1O3_mg = metal_edge_extender(C1Li1O3_mg)

            cls.C2H4_entry = None
            cls.LiEC_RO_entry = None
            cls.C1Li1O3_entry = None

            for entry in cls.LiEC_reextended_entries:
                if (
                    entry.formula == "C2 H4"
                    and entry.charge == 0
                    and entry.Nbonds == 5
                    and cls.C2H4_mg.isomorphic_to(entry.mol_graph)
                ):
                    if cls.C2H4_entry is not None:
                        if cls.C2H4_entry.free_energy() >= entry.free_energy():
                            cls.C2H4_entry = entry
                    else:
                        cls.C2H4_entry = entry

                if (
                    entry.formula == "C3 H4 Li1 O3"
                    and entry.charge == 0
                    and entry.Nbonds == 11
                    and cls.LiEC_RO_mg.isomorphic_to(entry.mol_graph)
                ):
                    if cls.LiEC_RO_entry is not None:
                        if cls.LiEC_RO_entry.free_energy() >= entry.free_energy():
                            cls.LiEC_RO_entry = entry
                    else:
                        cls.LiEC_RO_entry = entry

                if (
                    entry.formula == "C1 Li1 O3"
                    and entry.charge == 0
                    and entry.Nbonds == 5
                    and cls.C1Li1O3_mg.isomorphic_to(entry.mol_graph)
                ):
                    if cls.C1Li1O3_entry is not None:
                        if cls.C1Li1O3_entry.free_energy() >= entry.free_energy():
                            cls.C1Li1O3_entry = entry
                    else:
                        cls.C1Li1O3_entry = entry

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)

        # set up input variables
        C2H4_ind = None
        LiEC_RO_ind = None
        C1Li1O3_ind = None
        C2H4_RN_entry = None
        LiEC_RO_RN_entry = None
        C1Li1O3_RN_entry = None

        for entry in RN.entries["C2 H4"][5][0]:
            if self.C2H4_mg.isomorphic_to(entry.mol_graph):
                C2H4_ind = entry.parameters["ind"]
                C2H4_RN_entry = entry
                break
        for entry in RN.entries["C3 H4 Li1 O3"][11][0]:
            if self.LiEC_RO_mg.isomorphic_to(entry.mol_graph):
                LiEC_RO_ind = entry.parameters["ind"]
                LiEC_RO_RN_entry = entry
                break
        for entry in RN.entries["C1 Li1 O3"][5][0]:
            if self.C1Li1O3_mg.isomorphic_to(entry.mol_graph):
                C1Li1O3_ind = entry.parameters["ind"]
                C1Li1O3_RN_entry = entry
                break

        # perform calc
        reaction = IntermolecularReaction(LiEC_RO_RN_entry, [C2H4_RN_entry, C1Li1O3_RN_entry])
        graph = reaction.graph_representation()

        # assert
        self.assertCountEqual(
            list(graph.nodes),
            [
                LiEC_RO_ind,
                C2H4_ind,
                C1Li1O3_ind,
                str(LiEC_RO_ind) + "," + str(C1Li1O3_ind) + "+" + str(C2H4_ind),
                str(C2H4_ind) + "+PR_" + str(C1Li1O3_ind) + "," + str(LiEC_RO_ind),
                str(C1Li1O3_ind) + "+PR_" + str(C2H4_ind) + "," + str(LiEC_RO_ind),
            ],
        )
        self.assertEqual(len(graph.edges), 7)
        self.assertEqual(
            graph.get_edge_data(
                LiEC_RO_ind, str(LiEC_RO_ind) + "," + str(C1Li1O3_ind) + "+" + str(C2H4_ind),
            )["softplus"],
            0.5828092060367285,
        )
        self.assertEqual(
            graph.get_edge_data(
                LiEC_RO_ind, str(C2H4_ind) + "+PR_" + str(C1Li1O3_ind) + "," + str(LiEC_RO_ind),
            ),
            None,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = IntermolecularReaction.generate(RN.entries)

        self.assertEqual(len(reactions), 3673)

        for r in reactions:
            # if r.reactant.entry_id == self.LiEC_RO_entry.entry_id:
            #    if (
            #        r.products[0].entry_id == self.C2H4_entry.entry_id
            #        or r.products[1].entry_id == self.C2H4_entry.entry_id
            #    ):
            #        self.assertTrue(
            #            r.products[0].formula == "C1 Li1 O3" or r.products[1].formula == "C1 Li1 O3"
            #        )
            #        self.assertTrue(r.products[0].charge == 0 or r.products[1].charge == 0)
            #        self.assertTrue(
            #            r.products[0].free_energy() == self.C1Li1O3_entry.free_energy()
            #            or r.products[1].free_energy() == self.C1Li1O3_entry.free_energy()
            #        )
            pass

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        entries = bucket_mol_entries([self.LiEC_RO_entry, self.C1Li1O3_entry, self.C2H4_entry])

        reactions, families = IntermolecularReaction.generate(entries)
        self.assertEqual(len(reactions), 1)
        rxn = reactions[0]
        self.assertEqual(rxn.reactant_ids[0], self.LiEC_RO_entry.entry_id)
        self.assertEqual(rxn.product_ids[0], self.C2H4_entry.entry_id)
        self.assertEqual(rxn.product_ids[1], self.C1Li1O3_entry.entry_id)

        self.assertEqual(
            rxn.reactants_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )
        self.assertEqual(
            rxn.products_atom_mapping,
            [{0: 0, 1: 1, 2: 7, 3: 8, 4: 9, 5: 10}, {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = IntermolecularReaction(self.LiEC_RO_entry, [self.C1Li1O3_entry, self.C2H4_entry])
        reaction.free_energy()
        self.assertEqual(reaction.free_energy_A, 0.37075842588456)
        self.assertEqual(reaction.free_energy_B, -0.37075842588456)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = IntermolecularReaction(self.LiEC_RO_entry, [self.C1Li1O3_entry, self.C2H4_entry])
        reaction.energy()
        self.assertEqual(reaction.energy_A, 0.035409666514283344)
        self.assertEqual(reaction.energy_B, -0.035409666514283344)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = IntermolecularReaction(self.LiEC_RO_entry, [self.C1Li1O3_entry, self.C2H4_entry])
        self.assertEqual(reaction.rxn_type_A, "Molecular decomposition breaking one bond A -> B+C")
        self.assertEqual(reaction.rxn_type_B, "Molecular formation from one new bond A+B -> C")


class TestCoordinationBondChangeReaction(PymatgenTest):
    @classmethod
    def setUpClass(cls) -> None:
        if ob:
            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir, "LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(
                    molecule=mol, energy=E, enthalpy=H, entropy=S, entry_id=entry["task_id"],
                )
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
            )
            cls.EC_mg = metal_edge_extender(EC_mg)

            LiEC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
            )
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            cls.LiEC_entry = None
            cls.EC_minus_entry = None
            cls.Li_entry = None

            for entry in cls.LiEC_reextended_entries:
                if (
                    entry.formula == "C3 H4 O3"
                    and entry.charge == -1
                    and entry.Nbonds == 10
                    and cls.EC_mg.isomorphic_to(entry.mol_graph)
                ):
                    if cls.EC_minus_entry is not None:
                        if cls.EC_minus_entry.free_energy() >= entry.free_energy():
                            cls.EC_minus_entry = entry
                    else:
                        cls.EC_minus_entry = entry

                if (
                    entry.formula == "C3 H4 Li1 O3"
                    and entry.charge == 0
                    and entry.Nbonds == 12
                    and cls.LiEC_mg.isomorphic_to(entry.mol_graph)
                ):
                    if cls.LiEC_entry is not None:
                        if cls.LiEC_entry.free_energy() >= entry.free_energy():
                            cls.LiEC_entry = entry
                    else:
                        cls.LiEC_entry = entry

                if entry.formula == "Li1" and entry.charge == 1 and entry.Nbonds == 0:
                    if cls.Li_entry is not None:
                        if cls.Li_entry.free_energy() >= entry.free_energy():
                            cls.Li_entry = entry
                    else:
                        cls.Li_entry = entry

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)

        # set up input variables
        LiEC_ind = None
        EC_minus_ind = None
        Li_ind = None
        LiEC_RN_entry = None
        EC_minus_RN_entry = None
        Li_RN_entry = None

        for entry in RN.entries["C3 H4 Li1 O3"][12][0]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                LiEC_RN_entry = entry
                break
        for entry in RN.entries["C3 H4 O3"][10][-1]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_minus_ind = entry.parameters["ind"]
                EC_minus_RN_entry = entry
                break
        for entry in RN.entries["Li1"][0][1]:
            Li_ind = entry.parameters["ind"]
            Li_RN_entry = entry
            break

        # perform calc
        reaction = CoordinationBondChangeReaction(LiEC_RN_entry, [EC_minus_RN_entry, Li_RN_entry])
        graph = reaction.graph_representation()

        # assert
        self.assertCountEqual(
            list(graph.nodes),
            [
                LiEC_ind,
                EC_minus_ind,
                Li_ind,
                str(LiEC_ind) + "," + str(EC_minus_ind) + "+" + str(Li_ind),
                str(EC_minus_ind) + "+PR_" + str(Li_ind) + "," + str(LiEC_ind),
                str(Li_ind) + "+PR_" + str(EC_minus_ind) + "," + str(LiEC_ind),
            ],
        )
        self.assertEqual(len(graph.edges), 7)
        self.assertEqual(
            graph.get_edge_data(
                LiEC_ind, str(LiEC_ind) + "," + str(EC_minus_ind) + "+" + str(Li_ind)
            )["softplus"],
            1.5036425808336291,
        )
        self.assertEqual(
            graph.get_edge_data(
                LiEC_ind, str(Li_ind) + "+PR_" + str(EC_minus_ind) + "," + str(LiEC_ind)
            ),
            None,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = CoordinationBondChangeReaction.generate(RN.entries)
        self.assertEqual(len(reactions), 50)

        # for r in reactions:
        #    if r.reactant.entry_id == self.LiEC_entry.entry_id:
        #        if (
        #            r.products[0].entry_id == self.Li_entry.entry_id
        #            or r.products[1].entry_id == self.Li_entry.entry_id
        #        ):
        #            self.assertTrue(
        #                r.products[0].entry_id == self.EC_minus_entry.entry_id
        #                or r.products[1].entry_id == self.EC_minus_entry.entry_id
        #            )
        #            self.assertTrue(
        #                r.products[0].free_energy() == self.EC_minus_entry.free_energy()
        #                or r.products[1].free_energy() == self.EC_minus_entry.free_energy()
        #            )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        entries = bucket_mol_entries([self.LiEC_entry, self.EC_minus_entry, self.Li_entry])

        reactions, families = CoordinationBondChangeReaction.generate(entries)
        self.assertEqual(len(reactions), 1)
        rxn = reactions[0]
        self.assertEqual(rxn.reactant_ids[0], self.LiEC_entry.entry_id)
        self.assertEqual(rxn.product_ids[0], self.EC_minus_entry.entry_id)
        self.assertEqual(rxn.product_ids[1], self.Li_entry.entry_id)

        self.assertEqual(
            rxn.reactants_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )

        self.assertEqual(
            rxn.products_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9, 9: 10}, {0: 6}],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = CoordinationBondChangeReaction(
            self.LiEC_entry, [self.EC_minus_entry, self.Li_entry]
        )
        reaction.free_energy()
        self.assertEqual(reaction.free_energy_A, 1.857340187929367)
        self.assertEqual(reaction.free_energy_B, -1.857340187929367)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = CoordinationBondChangeReaction(
            self.LiEC_entry, [self.EC_minus_entry, self.Li_entry]
        )
        reaction.energy()
        self.assertEqual(reaction.energy_A, 0.08317397598398202)
        self.assertEqual(reaction.energy_B, -0.08317397598398202)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = CoordinationBondChangeReaction(
            self.LiEC_entry, [self.EC_minus_entry, self.Li_entry]
        )
        self.assertEqual(reaction.rxn_type_A, "Coordination bond breaking AM -> A+M")
        self.assertEqual(reaction.rxn_type_B, "Coordination bond forming A+M -> AM")


@unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
def test_bucket_mol_entries():
    C2H4_entry = MoleculeEntry(
        Molecule.from_file(os.path.join(test_dir, "C2H4.xyz")),
        energy=0.0,
        enthalpy=0.0,
        entropy=0.0,
    )
    LiEC_RO_entry = MoleculeEntry(
        Molecule.from_file(os.path.join(test_dir, "LiEC_RO.xyz")),
        energy=0.0,
        enthalpy=0.0,
        entropy=0.0,
    )
    C1Li1O3_entry = MoleculeEntry(
        Molecule.from_file(os.path.join(test_dir, "C1Li1O3.xyz")),
        energy=0.0,
        enthalpy=0.0,
        entropy=0.0,
    )

    bucket = bucket_mol_entries([C2H4_entry, LiEC_RO_entry, C1Li1O3_entry])

    ref_dict = {
        "C2 H4": {5: {0: [C2H4_entry]}},
        "C3 H4 Li1 O3": {11: {0: [LiEC_RO_entry]}},
        "C1 Li1 O3": {5: {0: [C1Li1O3_entry]}},
    }
    assert bucket == ref_dict


def test_unbucket_mol_entries():
    d = {"a": {"aa": [0, 1, 2], "aaa": [3, 4]}, "b": {"bb": [5, 6, 7], "bbb": (8, 9)}}
    out = unbucket_mol_entries(d)
    assert out == list(range(10))


if __name__ == "__main__":
    unittest.main()
