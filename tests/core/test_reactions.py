# coding: utf-8
import os
import unittest

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

from mrnet.core.reactions import (
    RedoxReaction,
    IntramolSingleBondChangeReaction,
    IntermolecularReaction,
    CoordinationBondChangeReaction,
    MetalHopReaction,
)
from mrnet.core.reactions import bucket_mol_entries, unbucket_mol_entries
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.network.reaction_network import ReactionNetwork

from monty.serialization import loadfn

try:
    import openbabel as ob
except ImportError:
    ob = None


test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)


def get_entries():
    if ob:
        LiEC_reextended_entries = []
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
                molecule=mol,
                energy=E,
                enthalpy=H,
                entropy=S,
                entry_id=entry["task_id"],
            )
            if mol_entry.formula == "Li1":
                if mol_entry.charge == 1:
                    LiEC_reextended_entries.append(mol_entry)
            else:
                LiEC_reextended_entries.append(mol_entry)

        RN = ReactionNetwork.from_input_entries(LiEC_reextended_entries)

        EC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
        )
        EC_mg = metal_edge_extender(EC_mg)

        LiEC_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
        )
        LiEC_mg = metal_edge_extender(LiEC_mg)

        LiEC_RO_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "LiEC_RO.xyz")), OpenBabelNN()
        )
        LiEC_RO_mg = metal_edge_extender(LiEC_RO_mg)

        C2H4_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "C2H4.xyz")), OpenBabelNN()
        )
        C2H4_mg = metal_edge_extender(C2H4_mg)

        C1Li1O3_mg = MoleculeGraph.with_local_env_strategy(
            Molecule.from_file(os.path.join(test_dir, "C1Li1O3.xyz")), OpenBabelNN()
        )
        C1Li1O3_mg = metal_edge_extender(C1Li1O3_mg)

        LiEC_entry = None
        LiEC_plus_entry = None
        EC_minus_entry = None
        EC_0_entry = None
        EC_1_entry = None
        LiEC_RO_entry = None
        C2H4_entry = None
        C1Li1O3_entry = None
        Li_entry = None

        for entry in RN.entries_list:
            if (
                entry.formula == "C3 H4 O3"
                and entry.num_bonds == 10
                and EC_mg.isomorphic_to(entry.mol_graph)
            ):
                if entry.charge == -1:
                    if EC_minus_entry is not None:
                        if EC_minus_entry.get_free_energy() >= entry.get_free_energy():
                            EC_minus_entry = entry
                    else:
                        EC_minus_entry = entry
                elif entry.charge == 0:
                    if EC_0_entry is not None:
                        if EC_0_entry.get_free_energy() >= entry.get_free_energy():
                            EC_0_entry = entry
                    else:
                        EC_0_entry = entry
                elif entry.charge == 1:
                    if EC_1_entry is not None:
                        if EC_1_entry.get_free_energy() >= entry.get_free_energy():
                            EC_1_entry = entry
                    else:
                        EC_1_entry = entry
            elif (
                entry.formula == "C3 H4 Li1 O3"
                and entry.num_bonds == 12
                and LiEC_mg.isomorphic_to(entry.mol_graph)
            ):
                if entry.charge == 0:
                    if LiEC_entry is not None:
                        if LiEC_entry.get_free_energy() >= entry.get_free_energy():
                            LiEC_entry = entry
                    else:
                        LiEC_entry = entry
                elif entry.charge == 1:
                    if LiEC_plus_entry is not None:
                        if LiEC_plus_entry.get_free_energy() >= entry.get_free_energy():
                            LiEC_plus_entry = entry
                    else:
                        LiEC_plus_entry = entry
            elif (
                entry.formula == "C3 H4 Li1 O3"
                and entry.charge == 0
                and entry.num_bonds == 11
                and LiEC_RO_mg.isomorphic_to(entry.mol_graph)
            ):
                if LiEC_RO_entry is not None:
                    if LiEC_RO_entry.get_free_energy() >= entry.get_free_energy():
                        LiEC_RO_entry = entry
                else:
                    LiEC_RO_entry = entry

            elif (
                entry.formula == "C2 H4"
                and entry.charge == 0
                and entry.num_bonds == 5
                and C2H4_mg.isomorphic_to(entry.mol_graph)
            ):
                if C2H4_entry is not None:
                    if C2H4_entry.get_free_energy() >= entry.get_free_energy():
                        C2H4_entry = entry
                else:
                    C2H4_entry = entry

            elif (
                entry.formula == "C1 Li1 O3"
                and entry.charge == 0
                and entry.num_bonds == 5
                and C1Li1O3_mg.isomorphic_to(entry.mol_graph)
            ):
                if C1Li1O3_entry is not None:
                    if C1Li1O3_entry.get_free_energy() >= entry.get_free_energy():
                        C1Li1O3_entry = entry
                else:
                    C1Li1O3_entry = entry
            elif entry.formula == "Li1" and entry.charge == 1 and entry.num_bonds == 0:
                if Li_entry is not None:
                    if Li_entry.get_free_energy() >= entry.get_free_energy():
                        Li_entry = entry
                else:
                    Li_entry = entry

        return {
            "entries": LiEC_reextended_entries,
            "RN": RN,
            "LiEC": LiEC_entry,
            "LiEC_plus": LiEC_plus_entry,
            "EC_1": EC_1_entry,
            "EC_0": EC_0_entry,
            "EC_-1": EC_minus_entry,
            "LiEC_RO": LiEC_RO_entry,
            "C2H4": C2H4_entry,
            "C1Li1O3": C1Li1O3_entry,
            "Li": Li_entry,
        }

    else:
        return None


entries = get_entries()


class TestRedoxReaction(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        reaction = RedoxReaction(entries["EC_0"], entries["EC_1"])
        reaction.electron_free_energy = -2.15
        graph = reaction.graph_representation()

        EC_0_ind = entries["EC_0"].parameters["ind"]
        EC_1_ind = entries["EC_1"].parameters["ind"]

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
            graph.get_edge_data(EC_0_ind, str(EC_0_ind) + "," + str(EC_1_ind))[
                "softplus"
            ],
            5.629805462349386,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        reactions = RedoxReaction.generate(entries["RN"].entries)

        self.assertEqual(len(reactions), 273)

        for r in reactions:
            if r.reactant == entries["EC_0"]:
                self.assertEqual(r.product.entry_id, entries["EC_1"].entry_id)
            if r.reactant == entries["EC_-1"]:
                self.assertEqual(r.product.entry_id, entries["EC_0"].entry_id)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):
        ents = bucket_mol_entries([entries["EC_-1"], entries["EC_0"], entries["EC_1"]])

        reactions = RedoxReaction.generate(ents)
        self.assertEqual(len(reactions), 2)

        for r in reactions:
            self.assertEqual(
                r.reactant_atom_mapping,
                [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}],
            )
            self.assertEqual(
                r.product_atom_mapping,
                [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}],
            )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):
        reaction = RedoxReaction(entries["EC_0"], entries["EC_1"])
        reaction.electron_free_energy = -2.15
        reaction.set_free_energy()
        self.assertEqual(reaction.free_energy_A, 6.231346035181195)
        self.assertEqual(reaction.free_energy_B, -6.231346035181195)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):
        reaction = RedoxReaction(entries["EC_0"], entries["EC_1"])
        reaction.electron_free_energy = -2.15
        self.assertEqual(reaction.energy_A, 0.3149076465170424)
        self.assertEqual(reaction.energy_B, -0.3149076465170424)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):
        reaction = RedoxReaction(entries["EC_0"], entries["EC_1"])
        self.assertEqual(reaction.__class__.__name__, "RedoxReaction")
        self.assertEqual(reaction.rxn_type_A, "One electron oxidation")
        self.assertEqual(reaction.rxn_type_B, "One electron reduction")


class TestIntramolSingleBondChangeReaction(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        LiEC_RN_entry = entries["LiEC"]
        LiEC_RO_RN_entry = entries["LiEC_RO"]
        LiEC_ind = LiEC_RN_entry.parameters["ind"]
        LiEC_RO_ind = LiEC_RO_RN_entry.parameters["ind"]

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
            graph.get_edge_data(LiEC_ind, str(LiEC_ind) + "," + str(LiEC_RO_ind))[
                "softplus"
            ],
            0.15092362164364986,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        reactions = IntramolSingleBondChangeReaction.generate(entries["RN"].entries)
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
            if r.reactant == entries["LiEC_RO"]:
                self.assertEqual(r.product.entry_id, entries["LiEC"].entry_id)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        ents = bucket_mol_entries([entries["LiEC_RO"], entries["LiEC"]])

        reactions = IntramolSingleBondChangeReaction.generate(ents)
        self.assertEqual(len(reactions), 1)
        rxn = reactions[0]
        self.assertEqual(rxn.reactant.entry_id, entries["LiEC_RO"].entry_id)
        self.assertEqual(rxn.product.entry_id, entries["LiEC"].entry_id)

        self.assertEqual(
            rxn.reactant_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )
        self.assertEqual(
            rxn.product_atom_mapping,
            [{0: 2, 1: 3, 2: 1, 3: 4, 4: 5, 5: 0, 6: 6, 7: 9, 8: 10, 9: 7, 10: 8}],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):
        reaction = IntramolSingleBondChangeReaction(entries["LiEC"], entries["LiEC_RO"])
        reaction.set_free_energy()
        self.assertEqual(reaction.free_energy_A, -1.2094343722765188)
        self.assertEqual(reaction.free_energy_B, 1.2094343722765188)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):
        reaction = IntramolSingleBondChangeReaction(entries["LiEC"], entries["LiEC_RO"])
        self.assertEqual(reaction.energy_A, -0.0377282729020294)
        self.assertEqual(reaction.energy_B, 0.0377282729020294)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = IntramolSingleBondChangeReaction(entries["LiEC"], entries["LiEC_RO"])
        self.assertEqual(
            reaction.__class__.__name__, "IntramolSingleBondChangeReaction"
        )
        self.assertEqual(reaction.rxn_type_A, "Intramolecular single bond formation")
        self.assertEqual(reaction.rxn_type_B, "Intramolecular single bond breakage")


class TestIntermolecularReaction(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        # set up input variables
        C2H4_RN_entry = entries["C2H4"]
        LiEC_RO_RN_entry = entries["LiEC_RO"]
        C1Li1O3_RN_entry = entries["C1Li1O3"]

        C2H4_ind = C2H4_RN_entry.parameters["ind"]
        LiEC_RO_ind = LiEC_RO_RN_entry.parameters["ind"]
        C1Li1O3_ind = C1Li1O3_RN_entry.parameters["ind"]

        # perform calc
        reaction = IntermolecularReaction(
            LiEC_RO_RN_entry, [C2H4_RN_entry, C1Li1O3_RN_entry]
        )
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
                LiEC_RO_ind,
                str(LiEC_RO_ind) + "," + str(C1Li1O3_ind) + "+" + str(C2H4_ind),
            )["softplus"],
            0.5828092060367285,
        )
        self.assertEqual(
            graph.get_edge_data(
                LiEC_RO_ind,
                str(C2H4_ind) + "+PR_" + str(C1Li1O3_ind) + "," + str(LiEC_RO_ind),
            ),
            None,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):
        reactions = IntermolecularReaction.generate(entries["RN"].entries)

        self.assertEqual(len(reactions), 3673)

        for r in reactions:
            if r.reactant.entry_id == entries["LiEC_RO"].entry_id:
                if (
                    r.products[0].entry_id == entries["C2H4"].entry_id
                    or r.products[1].entry_id == entries["C2H4"].entry_id
                ):
                    self.assertTrue(
                        r.products[0].formula == "C1 Li1 O3"
                        or r.products[1].formula == "C1 Li1 O3"
                    )
                    self.assertTrue(
                        r.products[0].get_free_energy()
                        == entries["C1Li1O3"].get_free_energy()
                        or r.products[1].get_free_energy()
                        == entries["C1Li1O3"].get_free_energy()
                    )
                    self.assertTrue(
                        r.products[0].get_free_energy()
                        == entries["C1Li1O3"].get_free_energy()
                        or r.products[1].get_free_energy()
                        == entries["C1Li1O3"].get_free_energy()
                    )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        ents = bucket_mol_entries(
            [entries["LiEC_RO"], entries["C1Li1O3"], entries["C2H4"]]
        )

        reactions = IntermolecularReaction.generate(ents)
        self.assertEqual(len(reactions), 1)
        rxn = reactions[0]
        self.assertEqual(rxn.reactant.entry_id, entries["LiEC_RO"].entry_id)
        self.assertEqual(rxn.product_0.entry_id, entries["C2H4"].entry_id)
        self.assertEqual(rxn.product_1.entry_id, entries["C1Li1O3"].entry_id)

        self.assertEqual(
            rxn.reactant_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )
        self.assertEqual(
            rxn.product_atom_mapping,
            [{0: 0, 1: 1, 2: 7, 3: 8, 4: 9, 5: 10}, {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = IntermolecularReaction(
            entries["LiEC_RO"], [entries["C1Li1O3"], entries["C2H4"]]
        )
        reaction.set_free_energy()
        self.assertEqual(reaction.free_energy_A, 0.37075842588456)
        self.assertEqual(reaction.free_energy_B, -0.37075842588410524)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = IntermolecularReaction(
            entries["LiEC_RO"], [entries["C1Li1O3"], entries["C2H4"]]
        )
        self.assertEqual(reaction.energy_A, 0.035409666514283344)
        self.assertEqual(reaction.energy_B, -0.035409666514283344)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = IntermolecularReaction(
            entries["LiEC_RO"], [entries["C1Li1O3"], entries["C2H4"]]
        )
        self.assertEqual(reaction.__class__.__name__, "IntermolecularReaction")
        self.assertEqual(
            reaction.rxn_type_A, "Molecular decomposition breaking one bond A -> B+C"
        )
        self.assertEqual(
            reaction.rxn_type_B, "Molecular formation from one new bond A+B -> C"
        )


class TestCoordinationBondChangeReaction(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        # set up input variables
        LiEC_RN_entry = entries["LiEC"]
        EC_minus_RN_entry = entries["EC_-1"]
        Li_RN_entry = entries["Li"]
        LiEC_ind = LiEC_RN_entry.parameters["ind"]
        EC_minus_ind = EC_minus_RN_entry.parameters["ind"]
        Li_ind = Li_RN_entry.parameters["ind"]

        # perform calc
        reaction = CoordinationBondChangeReaction(
            LiEC_RN_entry, [EC_minus_RN_entry, Li_RN_entry]
        )
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

        reactions = CoordinationBondChangeReaction.generate(entries["RN"].entries)
        self.assertEqual(len(reactions), 50)

        for r in reactions:
            if r.reactant.entry_id == entries["LiEC"].entry_id:
                if (
                    r.products[0].entry_id == entries["Li"].entry_id
                    or r.products[1].entry_id == entries["Li"].entry_id
                ):
                    self.assertTrue(
                        r.products[0].entry_id == entries["EC_-1"].entry_id
                        or r.products[1].entry_id == entries["EC_-1"].entry_id
                    )
                    self.assertTrue(
                        r.products[0].get_free_energy()
                        == entries["EC_-1"].get_free_energy()
                        or r.products[1].get_free_energy()
                        == entries["EC_-1"].get_free_energy()
                    )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_atom_mapping(self):

        ents = bucket_mol_entries([entries["LiEC"], entries["EC_-1"], entries["Li"]])

        reactions = CoordinationBondChangeReaction.generate(ents)
        self.assertEqual(len(reactions), 1)
        rxn = reactions[0]
        self.assertEqual(rxn.reactant.entry_id, entries["LiEC"].entry_id)
        self.assertEqual(rxn.product_0.entry_id, entries["EC_-1"].entry_id)
        self.assertEqual(rxn.product_1.entry_id, entries["Li"].entry_id)

        self.assertEqual(
            rxn.reactant_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}],
        )

        self.assertEqual(
            rxn.product_atom_mapping,
            [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9, 9: 10}, {0: 6}],
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = CoordinationBondChangeReaction(
            entries["LiEC"], [entries["EC_-1"], entries["Li"]]
        )
        reaction.set_free_energy()
        self.assertEqual(reaction.free_energy_A, 1.857340187929367)
        self.assertEqual(reaction.free_energy_B, -1.8573401879297649)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = CoordinationBondChangeReaction(
            entries["LiEC"], [entries["EC_-1"], entries["Li"]]
        )
        self.assertEqual(reaction.energy_A, 0.08317397598398202)
        self.assertEqual(reaction.energy_B, -0.08317397598399001)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = CoordinationBondChangeReaction(
            entries["LiEC"], [entries["EC_-1"], entries["Li"]]
        )
        self.assertEqual(reaction.__class__.__name__, "CoordinationBondChangeReaction")
        self.assertEqual(reaction.rxn_type_A, "Coordination bond breaking AM -> A+M")
        self.assertEqual(reaction.rxn_type_B, "Coordination bond forming A+M -> AM")


class TestMetalHopReaction(PymatgenTest):
    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        # perform calc
        reaction = MetalHopReaction(
            [entries["LiEC"], entries["EC_0"]],
            [entries["EC_-1"], entries["LiEC_plus"]],
            entries["Li"],
        )
        graph = reaction.graph_representation()

        liec_ind = entries["LiEC"].parameters["ind"]
        ec_ind = entries["EC_0"].parameters["ind"]
        liec_plus_ind = entries["LiEC_plus"].parameters["ind"]
        ec_minus_ind = entries["EC_-1"].parameters["ind"]

        # assert
        self.assertCountEqual(
            list(graph.nodes),
            [
                liec_ind,
                ec_ind,
                liec_plus_ind,
                ec_minus_ind,
                str(liec_ind)
                + "+PR_"
                + str(ec_ind)
                + ","
                + str(liec_plus_ind)
                + "+"
                + str(ec_minus_ind),
                str(ec_ind)
                + "+PR_"
                + str(liec_ind)
                + ","
                + str(liec_plus_ind)
                + "+"
                + str(ec_minus_ind),
                str(ec_minus_ind)
                + "+PR_"
                + str(liec_plus_ind)
                + ","
                + str(liec_ind)
                + "+"
                + str(ec_ind),
                str(liec_plus_ind)
                + "+PR_"
                + str(ec_minus_ind)
                + ","
                + str(liec_ind)
                + "+"
                + str(ec_ind),
            ],
        )
        self.assertEqual(len(graph.edges), 12)
        self.assertEqual(
            graph.get_edge_data(
                liec_ind,
                str(liec_ind)
                + "+PR_"
                + str(ec_ind)
                + ","
                + str(liec_plus_ind)
                + "+"
                + str(ec_minus_ind),
            )["softplus"],
            1.1019073858904995,
        )
        self.assertEqual(
            graph.get_edge_data(
                liec_ind,
                str(ec_minus_ind)
                + "+PR_"
                + str(liec_plus_ind)
                + ","
                + str(liec_ind)
                + "+"
                + str(ec_ind),
            ),
            None,
        )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        reactions = MetalHopReaction.generate(entries["RN"].entries)
        self.assertEqual(len(reactions), 4753)

        for r in reactions:
            if (
                r.reactants[0].entry_id == entries["LiEC"].entry_id
                and r.reactants[1].entry_id == entries["EC_0"].entry_id
            ):
                if (
                    r.products[0].entry_id == entries["EC_-1"].entry_id
                    or r.products[1].entry_id == entries["EC_-1"].entry_id
                ):
                    self.assertTrue(
                        r.products[0].entry_id == entries["LiEC_plus"].entry_id
                        or r.products[1].entry_id == entries["LiEC_plus"].entry_id
                    )
                    self.assertTrue(
                        r.products[0].get_free_energy()
                        == entries["LiEC_plus"].get_free_energy()
                        or r.products[1].get_free_energy()
                        == entries["LiEC_plus"].get_free_energy()
                    )

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = MetalHopReaction(
            [entries["LiEC"], entries["EC_0"]],
            [entries["EC_-1"], entries["LiEC_plus"]],
            entries["Li"],
        )
        reaction.set_free_energy()
        self.assertEqual(reaction.free_energy_A, 1.303222066930175)
        self.assertEqual(reaction.free_energy_B, -1.303222066930175)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = MetalHopReaction(
            [entries["LiEC"], entries["EC_0"]],
            [entries["EC_-1"], entries["LiEC_plus"]],
            entries["Li"],
        )
        self.assertEqual(reaction.energy_A, 0.051295952076088724)
        self.assertEqual(reaction.energy_B, -0.051295952076088724)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = MetalHopReaction(
            [entries["LiEC"], entries["EC_0"]],
            [entries["EC_-1"], entries["LiEC_plus"]],
            entries["Li"],
        )
        self.assertEqual(reaction.__class__.__name__, "MetalHopReaction")
        self.assertEqual(
            reaction.rxn_type_A, "Metal hopping reaction AM + B <-> A + BM"
        )
        self.assertEqual(
            reaction.rxn_type_B, "Metal hopping reaction AM + B <-> A + BM"
        )


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
