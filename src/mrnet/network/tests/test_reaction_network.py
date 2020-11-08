# coding: utf-8
import io
import os
import sys
import unittest
import copy
import pickle

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender

from mrnet.network.reaction_network import (RedoxReaction,
                                            IntramolSingleBondChangeReaction,
                                            IntermolecularReaction,
                                            CoordinationBondChangeReaction,
                                            ReactionPath,
                                            ReactionNetwork)

from monty.serialization import dumpfn, loadfn
from networkx.readwrite import json_graph

try:
    import openbabel as ob
except ImportError:
    ob = None

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        'test_files', 'reaction_network_files')


class TestRedoxReaction(PymatgenTest):

    @classmethod
    def setUpClass(cls) -> None:
        if ob:
            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir,"LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(molecule=mol, energy=E, enthalpy=H,
                                          entropy=S, entry_id=entry["task_id"])
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "EC.xyz")),
                OpenBabelNN())
            cls.EC_mg = metal_edge_extender(EC_mg)\

            cls.EC_0_entry = None
            cls.EC_minus_entry = None
            cls.EC_1_entry = None

            for entry in cls.LiEC_reextended_entries:
                if entry.formula == "C3 H4 O3" and entry.charge == 0 and entry.Nbonds == 10 \
                        and cls.EC_mg.isomorphic_to(entry.mol_graph):
                    cls.EC_0_entry = entry
                elif entry.formula == "C3 H4 O3" and entry.charge == -1 and entry.Nbonds == 10 \
                        and cls.EC_mg.isomorphic_to(entry.mol_graph):
                    cls.EC_minus_entry = entry
                elif entry.formula == "C3 H4 O3" and entry.charge == 1 and entry.Nbonds == 10 \
                        and cls.EC_mg.isomorphic_to(
                        entry.mol_graph):
                    cls.EC_1_entry = entry
                if cls.EC_0_entry is not None and cls.EC_minus_entry is not None \
                        and cls.EC_1_entry is not None:
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

        self.assertCountEqual(list(graph.nodes), [EC_0_ind, EC_1_ind, str(EC_0_ind) + "," + str(EC_1_ind),
                                                  str(EC_1_ind) + "," + str(EC_0_ind)])
        self.assertEqual(len(graph.edges), 4)
        self.assertEqual(graph.get_edge_data(EC_0_ind,
                                             str(EC_0_ind) + "," + str(EC_1_ind))["softplus"],
                         5.629805462349386)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = RedoxReaction.generate(RN.entries)

        self.assertEqual(len(reactions), 273)

        for r in reactions:
            if r.reactant == self.EC_0_entry:
                self.assertEqual(r.product.entry_id, self.EC_1_entry.entry_id)
            if r.reactant == self.EC_minus_entry:
                self.assertEqual(r.product.entry_id, self.EC_0_entry.entry_id)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = RedoxReaction(self.EC_0_entry, self.EC_1_entry)
        reaction.electron_free_energy = -2.15
        free_energy_dict = reaction.free_energy()
        self.assertEqual(free_energy_dict, {'free_energy_A': 6.231346035181195,
                                            'free_energy_B': -6.231346035181195})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = RedoxReaction(self.EC_0_entry, self.EC_1_entry)
        reaction.electron_free_energy = -2.15
        energy_dict = reaction.energy()
        self.assertEqual(energy_dict, {'energy_A': 0.3149076465170424,
                                       'energy_B': -0.3149076465170424})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = RedoxReaction(self.EC_0_entry, self.EC_1_entry)
        type_dict = reaction.reaction_type()
        self.assertEqual(type_dict, {'class': 'RedoxReaction',
                                     'rxn_type_A': 'One electron oxidation',
                                     'rxn_type_B': 'One electron reduction'})


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
                mol_entry = MoleculeEntry(molecule=mol, energy=E, enthalpy=H,
                                          entropy=S, entry_id=entry["task_id"])
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            LiEC_mg =  MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir,"LiEC.xyz")),
                OpenBabelNN())
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            LiEC_RO_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC_RO.xyz")),
                OpenBabelNN())
            cls.LiEC_RO_mg = metal_edge_extender(LiEC_RO_mg)

            cls.LiEC_entry = None
            cls.LiEC_RO_entry = None

            for entry in cls.LiEC_reextended_entries:
                if entry.formula == "C3 H4 Li1 O3" and entry.charge == 0 \
                        and entry.Nbonds == 12 and cls.LiEC_mg.isomorphic_to(
                        entry.mol_graph):
                    cls.LiEC_entry = entry
                elif entry.formula == "C3 H4 Li1 O3" and entry.charge == 0 \
                        and entry.Nbonds == 11 and cls.LiEC_RO_mg.isomorphic_to(
                        entry.mol_graph):
                    cls.LiEC_RO_entry = entry
                if cls.LiEC_entry is not None and cls.LiEC_RO_entry is not None:
                    break

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_graph_representation(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        #print(RN.entries["C3 H4 Li1 O3"][11][0][0].molecule)

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
        reaction = IntramolSingleBondChangeReaction(LiEC_RN_entry,
                                                    LiEC_RO_RN_entry)
        reaction.electron_free_energy = -2.15
        graph = reaction.graph_representation()
        print(graph.nodes, graph.edges)
        print(graph.get_edge_data(LiEC_ind, str(LiEC_ind) + "," + str(LiEC_RO_ind)))
        self.assertCountEqual(list(graph.nodes), [LiEC_ind, LiEC_RO_ind,
                                                  str(LiEC_ind) + "," + str(LiEC_RO_ind),
                                                  str(LiEC_RO_ind)+","+str(LiEC_ind)])
        self.assertEqual(len(graph.edges), 4)
        self.assertEqual(graph.get_edge_data(LiEC_ind,
                                             str(LiEC_ind) + "," + str(LiEC_RO_ind))["softplus"],
                         0.15092362164364986)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = IntramolSingleBondChangeReaction.generate(RN.entries)
        self.assertEqual(len(reactions), 93)

        for r in reactions:
            if r.reactant == self.LiEC_entry:
                self.assertEqual(r.product.entry_id, self.LiEC_RO_entry.entry_id)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = IntramolSingleBondChangeReaction(self.LiEC_entry, self.LiEC_RO_entry)
        free_energy_dict = reaction.free_energy()
        self.assertEqual(free_energy_dict, {'free_energy_A': -1.1988634269218892,
                                            'free_energy_B': 1.1988634269218892})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = IntramolSingleBondChangeReaction(self.LiEC_entry, self.LiEC_RO_entry)
        energy_dict = reaction.energy()
        self.assertEqual(energy_dict, {'energy_A': -0.03746218086303088,
                                       'energy_B': 0.03746218086303088})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = IntramolSingleBondChangeReaction(self.LiEC_entry, self.LiEC_RO_entry)
        type_dict = reaction.reaction_type()
        self.assertEqual(type_dict, {'class': 'IntramolSingleBondChangeReaction',
                                     'rxn_type_A': 'Intramolecular single bond formation',
                                     'rxn_type_B': 'Intramolecular single bond breakage'})


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
                mol_entry = MoleculeEntry(molecule=mol, energy=E, enthalpy=H,
                                          entropy=S, entry_id=entry["task_id"])
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            C2H4_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "C2H4.xyz")),
                OpenBabelNN())
            cls.C2H4_mg = metal_edge_extender(C2H4_mg)

            LiEC_RO_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LiEC_RO.xyz")),
                OpenBabelNN())
            cls.LiEC_RO_mg = metal_edge_extender(LiEC_RO_mg)

            C1Li1O3_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "C1Li1O3.xyz")),
                OpenBabelNN())
            cls.C1Li1O3_mg = metal_edge_extender(C1Li1O3_mg)

            cls.C2H4_entry = None
            cls.LiEC_RO_entry = None
            cls.C1Li1O3_entry = None

            for entry in cls.LiEC_reextended_entries:
                if entry.formula == "C2 H4" and entry.charge == 0 \
                        and entry.Nbonds == 5 and cls.C2H4_mg.isomorphic_to(
                        entry.mol_graph):
                    if cls.C2H4_entry is not None:
                        if cls.C2H4_entry.free_energy() >= entry.free_energy():
                            cls.C2H4_entry = entry
                    else:
                        cls.C2H4_entry = entry

                if entry.formula == "C3 H4 Li1 O3" and entry.charge == 0 \
                        and entry.Nbonds == 11 and cls.LiEC_RO_mg.isomorphic_to(
                        entry.mol_graph):
                    if cls.LiEC_RO_entry is not None:
                        if cls.LiEC_RO_entry.free_energy() >= entry.free_energy():
                            cls.LiEC_RO_entry = entry
                    else:
                        cls.LiEC_RO_entry = entry

                if entry.formula == "C1 Li1 O3" and entry.charge == 0 \
                        and entry.Nbonds == 5 and cls.C1Li1O3_mg.isomorphic_to(
                        entry.mol_graph):
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
        reaction = IntermolecularReaction(LiEC_RO_RN_entry, [C2H4_RN_entry,
                                                             C1Li1O3_RN_entry])
        graph = reaction.graph_representation()

        # assert
        self.assertCountEqual(list(graph.nodes), [LiEC_RO_ind, C2H4_ind,C1Li1O3_ind,
                                                  str(LiEC_RO_ind) + "," + str(C1Li1O3_ind) + "+" + str(C2H4_ind),
                                                  str(C2H4_ind) + "+PR_" + str(C1Li1O3_ind) + "," + str(LiEC_RO_ind),
                                                  str(C1Li1O3_ind) + "+PR_" + str(C2H4_ind) + "," + str(LiEC_RO_ind)])
        self.assertEqual(len(graph.edges), 7)
        self.assertEqual(graph.get_edge_data(LiEC_RO_ind,
                                             str(LiEC_RO_ind) + "," + str(C1Li1O3_ind) + "+"
                                             + str(C2H4_ind))["softplus"],
                         0.5828092060367285)
        self.assertEqual(graph.get_edge_data(LiEC_RO_ind, str(C2H4_ind) + "+PR_" + str(C1Li1O3_ind) + ","
                                             + str(LiEC_RO_ind)),
                         None)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = IntermolecularReaction.generate(RN.entries)

        self.assertEqual(len(reactions), 3673)

        for r in reactions:
            if r.reactant.entry_id == self.LiEC_RO_entry.entry_id:
                if r.products[0].entry_id == self.C2H4_entry.entry_id \
                        or r.products[0].entry_id == self.C2H4_entry.entry_id:
                    self.assertTrue(r.products[0].formula == "C1 Li1 O3" or r.products[1].formula == "C1 Li1 O3")
                    self.assertTrue(r.products[0].charge == 0 or r.products[1].charge == 0)
                    self.assertTrue(r.products[0].free_energy() == self.C1Li1O3_entry.free_energy()
                                    or r.products[1].free_energy() == self.C1Li1O3_entry.free_energy())

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = IntermolecularReaction(self.LiEC_RO_entry,
                                          [self.C1Li1O3_entry, self.C2H4_entry])
        free_energy_dict = reaction.free_energy()
        self.assertEqual(free_energy_dict, {'free_energy_A': 0.37075842588456,
                                            'free_energy_B': -0.37075842588410524})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = IntermolecularReaction(self.LiEC_RO_entry,
                                          [self.C1Li1O3_entry, self.C2H4_entry])
        energy_dict = reaction.energy()
        self.assertEqual(energy_dict, {'energy_A': 0.035409666514283344,
                                       'energy_B': -0.035409666514283344})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = IntermolecularReaction(self.LiEC_RO_entry,
                                          [self.C1Li1O3_entry, self.C2H4_entry])
        type_dict = reaction.reaction_type()
        self.assertEqual(type_dict, {'class': 'IntermolecularReaction',
                                     'rxn_type_A': 'Molecular decomposition breaking one bond A -> B+C',
                                     'rxn_type_B': 'Molecular formation from one new bond A+B -> C'})


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
                mol_entry = MoleculeEntry(molecule=mol, energy=E, enthalpy=H,
                                          entropy=S, entry_id=entry["task_id"])
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)

            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir,"EC.xyz")),
                OpenBabelNN())
            cls.EC_mg = metal_edge_extender(EC_mg)

            LiEC_mg =  MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir,"LiEC.xyz")),
                OpenBabelNN())
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            cls.LiEC_entry = None
            cls.EC_minus_entry = None
            cls.Li_entry = None

            for entry in cls.LiEC_reextended_entries:
                if entry.formula == "C3 H4 O3" and entry.charge == -1 \
                        and entry.Nbonds == 10 and cls.EC_mg.isomorphic_to(
                        entry.mol_graph):
                    if cls.EC_minus_entry is not None:
                        if cls.EC_minus_entry.free_energy() >= entry.free_energy():
                            cls.EC_minus_entry = entry
                    else:
                        cls.EC_minus_entry = entry

                if entry.formula == "C3 H4 Li1 O3" and entry.charge == 0 \
                        and entry.Nbonds == 12 and cls.LiEC_mg.isomorphic_to(
                        entry.mol_graph):
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
        reaction = CoordinationBondChangeReaction(LiEC_RN_entry,
                                                  [EC_minus_RN_entry, Li_RN_entry])
        graph = reaction.graph_representation()

        # assert
        self.assertCountEqual(list(graph.nodes), [LiEC_ind, EC_minus_ind, Li_ind,
                                                  str(LiEC_ind) + "," + str(EC_minus_ind) + "+" + str(Li_ind),
                                                  str(EC_minus_ind) + "+PR_" + str(Li_ind) + "," + str(LiEC_ind),
                                                  str(Li_ind) + "+PR_" + str(EC_minus_ind) + "," + str(LiEC_ind)])
        self.assertEqual(len(graph.edges), 7)
        self.assertEqual(
            graph.get_edge_data(LiEC_ind, str(LiEC_ind) + "," + str(EC_minus_ind) + "+" + str(Li_ind))[
                "softplus"], 1.5036425808336291)
        self.assertEqual(
            graph.get_edge_data(LiEC_ind, str(Li_ind) + "+PR_" + str(EC_minus_ind) + "," + str(LiEC_ind)), None)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_generate(self):

        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        reactions, families = CoordinationBondChangeReaction.generate(RN.entries)
        self.assertEqual(len(reactions), 50)

        for r in reactions:
            if r.reactant.entry_id == self.LiEC_entry.entry_id:
                if r.products[0].entry_id == self.Li_entry.entry_id or r.products[1].entry_id == self.Li_entry.entry_id:
                    self.assertTrue(r.products[0].entry_id == self.EC_minus_entry.entry_id
                                    or r.products[1].entry_id == self.EC_minus_entry.entry_id)
                    self.assertTrue(r.products[0].free_energy() == self.EC_minus_entry.free_energy()
                                    or r.products[1].free_energy() == self.EC_minus_entry.free_energy())

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_free_energy(self):

        reaction = CoordinationBondChangeReaction(self.LiEC_entry, [self.EC_minus_entry,
                                                                    self.Li_entry])
        free_energy_dict = reaction.free_energy()
        self.assertEqual(free_energy_dict, {'free_energy_A': 1.857340187929367,
                                            'free_energy_B': -1.8573401879297649})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_energy(self):

        reaction = CoordinationBondChangeReaction(self.LiEC_entry, [self.EC_minus_entry,
                                                                    self.Li_entry])
        energy_dict = reaction.energy()
        self.assertEqual(energy_dict, {'energy_A': 0.08317397598398202,
                                       'energy_B': -0.08317397598399001})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_reaction_type(self):

        reaction = CoordinationBondChangeReaction(self.LiEC_entry, [self.EC_minus_entry,
                                                                    self.Li_entry])
        type_dict = reaction.reaction_type()
        self.assertEqual(type_dict, {'class': 'CoordinationBondChangeReaction',
                                     'rxn_type_A': 'Coordination bond breaking AM -> A+M',
                                     'rxn_type_B': 'Coordination bond forming A+M -> AM'})


class TestReactionPath(PymatgenTest):

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_characterize_path(self):

        # set up input variables
        path = loadfn(os.path.join(test_dir, "characterize_path_path_IN.json"))
        graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir, "characterize_path_self_graph_IN.json")))
        self_min_cost_str = loadfn(os.path.join(test_dir, "characterize_path_self_min_cost_IN.json"))
        solved_PRs = loadfn(os.path.join(test_dir, "characterize_path_old_solved_PRs_IN.json"))
        PR_paths_str = loadfn(os.path.join(test_dir, "characterize_path_final_PR_paths_IN.json"))
        loaded_PR_byproducts = loadfn(os.path.join(test_dir, "PR_byproducts_dict.json"))

        PR_byproducts = {}
        for node in loaded_PR_byproducts:
            PR_byproducts[int(node)] = loaded_PR_byproducts[node]
        PR_paths = {}
        for node in PR_paths_str:
            PR_paths[int(node)] = {}
            for start in PR_paths_str[node]:
                PR_paths[int(node)][int(start)] = copy.deepcopy(PR_paths_str[node][start])

        self_min_cost = {}
        for node in self_min_cost_str:
            self_min_cost[int(node)] = self_min_cost_str[node]

        # run calc
        path_instance = ReactionPath.characterize_path(path, "softplus",
                                                       self_min_cost, graph,
                                                       solved_PRs,
                                                       PR_byproducts, PR_paths)

        # assert
        self.assertEqual(path_instance.byproducts, [456, 34])
        self.assertEqual(path_instance.unsolved_prereqs, [563, 250, 565, 544, 0, 564, 564])
        self.assertEqual(path_instance.solved_prereqs, [556])
        self.assertEqual(path_instance.cost, 1.0716192089248349)
        self.assertEqual(path_instance.pure_cost, 0.0)
        self.assertEqual(path_instance.hardest_step_deltaG, None)
        self.assertEqual(path_instance.path, [456, '456+PR_556,424', 424, '424,456+556', 556, '556+PR_563,558', 558,
                                              '558+PR_250,221', 221, '221+PR_565,232', 232, '232,34+83', 83,
                                              '83+PR_544,131', 131, '131,129', 129, '129+PR_0,310', 310,
                                              '310+PR_564,322', 322, '322+PR_564,333', 333],)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_characterize_path_final(self):

        #set up input variables
        path = loadfn(os.path.join(test_dir, "characterize_path_final_path_IN.json"))
        graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir,
                                                               "characterize_path_final_self_graph_IN.json")))
        self_min_cost_str = loadfn(os.path.join(test_dir,
                                                "characterize_path_final_self_min_cost_IN.json"))
        old_solved_PRs = loadfn(os.path.join(test_dir,
                                             "solved_PRs_list.json"))
        PR_paths_str = loadfn(os.path.join(test_dir,
                                           "characterize_path_final_PR_paths_IN.json"))
        loaded_PR_byproducts = loadfn(os.path.join(test_dir,
                                                   "PR_byproducts_dict.json"))

        PR_byproducts = {}
        for node in loaded_PR_byproducts:
            PR_byproducts[int(node)] = loaded_PR_byproducts[node]

        self_min_cost = {}
        for node in self_min_cost_str:
            self_min_cost[int(node)] = self_min_cost_str[node]

        PR_paths = {}
        for node in PR_paths_str:
            PR_paths[int(node)] = {}
            for start in PR_paths_str[node]:
                PR_paths[int(node)][int(start)] = copy.deepcopy(PR_paths_str[node][start])

        # perform calc
        path_class = ReactionPath.characterize_path_final(path, "softplus",
                                                          self_min_cost, graph,
                                                          old_solved_PRs,
                                                          PR_byproducts,
                                                          PR_paths)

        print(path_class.path_dict)

        # assert
        self.assertEqual(path_class.byproducts, [164])
        self.assertEqual(path_class.solved_prereqs, [51, 420])
        self.assertEqual(path_class.all_prereqs, [51, 420])
        self.assertEqual(path_class.cost, 2.6460023352176423)
        self.assertEqual(path_class.path, [556, '556+PR_51,41', 41, '41+PR_420,511', 511])
        self.assertEqual(path_class.overall_free_energy_change, -6.240179642712474)
        self.assertEqual(path_class.pure_cost, 2.6460023352176427)
        self.assertEqual(path_class.hardest_step_deltaG, 1.2835689714924228)


class TestReactionNetwork(PymatgenTest):

    @classmethod
    def setUpClass(cls):
        if ob:
            EC_mg = MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir,"EC.xyz")),
                OpenBabelNN())
            cls.EC_mg = metal_edge_extender(EC_mg)

            LiEC_mg =  MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir,"LiEC.xyz")),
                OpenBabelNN())
            cls.LiEC_mg = metal_edge_extender(LiEC_mg)

            LEDC_mg =  MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir,"LEDC.xyz")),
                OpenBabelNN())
            cls.LEDC_mg = metal_edge_extender(LEDC_mg)

            LEMC_mg =  MoleculeGraph.with_local_env_strategy(
                Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")),
                OpenBabelNN())
            cls.LEMC_mg = metal_edge_extender(LEMC_mg)

            cls.LiEC_reextended_entries = []
            entries = loadfn(os.path.join(test_dir,"LiEC_reextended_entries.json"))
            for entry in entries:
                if "optimized_molecule" in entry["output"]:
                    mol = entry["output"]["optimized_molecule"]
                else:
                    mol = entry["output"]["initial_molecule"]
                E = float(entry["output"]["final_energy"])
                H = float(entry["output"]["enthalpy"])
                S = float(entry["output"]["entropy"])
                mol_entry = MoleculeEntry(molecule=mol, energy=E, enthalpy=H,
                                          entropy=S ,entry_id=entry["task_id"])
                if mol_entry.formula == "Li1":
                    if mol_entry.charge == 1:
                        cls.LiEC_reextended_entries.append(mol_entry)
                else:
                    cls.LiEC_reextended_entries.append(mol_entry)
            dumpfn(cls.LiEC_reextended_entries, "unittest_input_molentries.json")
            cls.RN_cls = loadfn(os.path.join(test_dir, "RN.json"))

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_add_reactions(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries,
                                                electron_free_energy=-2.15)

        # set up input variables
        EC_0_entry = None
        EC_minus_entry = None

        # print(RN.entries["C3 H4 O3"].keys())

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_0_entry = entry
                break
        for entry in RN.entries["C3 H4 O3"][10][-1]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_minus_entry = entry
                break

        redox = RedoxReaction(EC_0_entry, EC_minus_entry)
        redox.electron_free_energy = -2.15
        redox_graph = redox.graph_representation()

        # run calc
        RN.add_reaction(redox_graph)

        # assert
        self.assertEqual(list(RN.graph.nodes), ['456,455', 456, 455, '455,456'])
        self.assertEqual(list(RN.graph.edges), [('456,455', 455), (456, '456,455'),
                                                (455, '455,456'), ('455,456', 456)])

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries,
                                                electron_free_energy=-2.15)

        # perfrom calc
        RN.build()

        # assert
        EC_ind = None
        LEDC_ind = None
        LiEC_ind = None
        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        self.assertEqual(len(RN.entries_list),569)
        self.assertEqual(EC_ind,456)
        self.assertEqual(LEDC_ind,511)
        self.assertEqual(Li1_ind,556)
        self.assertEqual(LiEC_ind,424)

        self.assertEqual(len(RN.graph.nodes),10481)
        self.assertEqual(len(RN.graph.edges),22890)

        # dumpfn(RN, os.path.join(test_dir, "RN.json"))

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build_PR_record(self):
        # set up RN
        RN = self.RN_cls
        RN.build()

        # run calc
        PR_record = RN.build_PR_record()

        # assert
        self.assertEqual(len(PR_record[0]), 42)
        self.assertEqual(PR_record[44], ['165+PR_44,434'])
        self.assertEqual(len(PR_record[529]), 0)
        self.assertEqual(len(PR_record[556]), 104)
        self.assertEqual(len(PR_record[564]), 165)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_build_reactant_record(self):

        # set up RN
        RN = self.RN_cls
        RN.build()

        # run calc
        reactant_record = RN.build_reactant_record()

        # assert
        self.assertEqual(len(reactant_record[0]), 43)
        self.assertCountEqual(reactant_record[44], ['44+PR_165,434', '44,43',
                                                    '44,40+556'])
        self.assertEqual(len(reactant_record[529]), 0)
        self.assertEqual(len(reactant_record[556]), 104)
        self.assertEqual(len(reactant_record[564]), 167)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_solve_prerequisites(self):

        # set up RN
        RN = self.RN_cls
        RN.build_PR_record()
        # set up input variables

        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        # perfrom calc
        PRs_calc, old_solved_PRs = RN.solve_prerequisites([EC_ind, Li1_ind],
                                                          weight="softplus")

        # with open(os.path.join(test_dir,"xx_PRs_unittest.json"), 'wb') as handle:
        #     pickle.dump(PRs_calc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # assert
        with open(os.path.join(test_dir,"xx_PRs_unittest.json"), 'rb') as handle:
            loaded_PRs = pickle.load(handle)

        #loaded_PRs = loadfn(PRs_filename)
        PR_paths = {}
        for key in loaded_PRs:
            PR_paths[int(key)] = {}
            for start in loaded_PRs[key]:
                PR_paths[int(key)][int(start)] = copy.deepcopy(loaded_PRs[key][start])

        for node in PRs_calc:
            for start in PRs_calc[node]:
                self.assertEqual(
                    [PRs_calc[node][start].all_prereqs,PRs_calc[node][start].byproducts,
                     PRs_calc[node][start].full_path, PRs_calc[node][start].path,
                     PRs_calc[node][start].solved_prereqs, PRs_calc[node][start].unsolved_prereqs],
                    [PR_paths[node][start].all_prereqs,PR_paths[node][start].byproducts,
                     PR_paths[node][start].full_path , PR_paths[node][start].path,
                     PR_paths[node][start].solved_prereqs, PR_paths[node][start].unsolved_prereqs])

                if PRs_calc[node][start].cost != PR_paths[node][start].cost:
                    self.assertAlmostEqual(PRs_calc[node][start].cost, PR_paths[node][start].cost, places=2)
                if PRs_calc[node][start].pure_cost != PR_paths[node][start].pure_cost:
                    self.assertAlmostEqual(PRs_calc[node][start].pure_cost,PR_paths[node][start].pure_cost, places=2)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_path_cost(self):

        # set up RN
        RN = self.RN_cls
        RN.weight = "softplus"
        RN.graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir,
                                                                  "find_path_cost_self_graph_IN.json")))
        loaded_self_min_cost_str = loadfn(os.path.join(test_dir,
                                                       "find_path_cost_self_min_cost_IN.json"))
        loaded_PR_byproducts = loadfn(os.path.join(test_dir,
                                                   "PR_byproducts_dict.json"))

        RN.PR_byproducts = {}
        for node in loaded_PR_byproducts:
            RN.PR_byproducts[int(node)] = loaded_PR_byproducts[node]

        for node in loaded_self_min_cost_str:
            RN.min_cost[int(node)] = loaded_self_min_cost_str[node]

        # set up input variables
        EC_ind = None
        LEDC_ind = None
        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        loaded_cost_from_start_str = loadfn(os.path.join(test_dir,
                                                         "find_path_cost_cost_from_start_IN.json"))
        old_solved_PRs = loadfn(os.path.join(test_dir,
                                             "find_path_cost_old_solved_PRs_IN.json"))
        loaded_min_cost_str = loadfn(os.path.join(test_dir,
                                                  "find_path_cost_min_cost_IN.json"))
        loaded_PRs_str = loadfn(os.path.join(test_dir,
                                             "find_path_cost_PRs_IN.json"))

        loaded_cost_from_start = {}
        for node in loaded_cost_from_start_str:
            loaded_cost_from_start[int(node)] = {}
            for start in loaded_cost_from_start_str[node]:
                loaded_cost_from_start[int(node)][int(start)] = loaded_cost_from_start_str[node][start]

        loaded_min_cost = {}
        for node in loaded_min_cost_str:
            loaded_min_cost[int(node)] = loaded_min_cost_str[node]

        loaded_PRs = {}
        for node in loaded_PRs_str:
            loaded_PRs[int(node)] = {}
            for start in loaded_PRs_str[node]:
                loaded_PRs[int(node)][int(start)] = copy.deepcopy(loaded_PRs_str[node][start])

        # perform calc
        PRs_cal, cost_from_start_cal, min_cost_cal = RN.find_path_cost([EC_ind, Li1_ind], RN.weight,
                                                                       old_solved_PRs,
                                                                       loaded_cost_from_start,
                                                                       loaded_min_cost, loaded_PRs)

        # assert
        self.assertEqual(cost_from_start_cal[456][456], 0.0)
        self.assertEqual(cost_from_start_cal[556][456], "no_path")
        self.assertEqual(cost_from_start_cal[0][456], 2.0148202484602122)
        self.assertEqual(cost_from_start_cal[6][556], 0.06494386469823213)
        self.assertEqual(cost_from_start_cal[80][456], 1.0882826020202816)

        self.assertEqual(min_cost_cal[556], 0.0)
        self.assertEqual(min_cost_cal[1], 0.9973160537476341)
        self.assertEqual(min_cost_cal[4], 0.2456832817986014)
        self.assertEqual(min_cost_cal[148], 0.09651432795671926)

        self.assertEqual(PRs_cal[556][556].path, [556])
        self.assertEqual(PRs_cal[556][456].path, None)
        self.assertEqual(PRs_cal[29][456].path, None)
        self.assertEqual(PRs_cal[313], {})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_identify_solved_PRs(self):

        # set up RN
        RN = self.RN_cls
        RN.num_starts = 2
        RN.weight = "softplus"
        RN.graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir,
                                                                  "identify_solved_PRs_self_graph_IN.json")))
        loaded_self_min_cost_str = loadfn(os.path.join(test_dir,
                                                       "identify_solved_PRs_self_min_cost_IN.json"))
        for node in loaded_self_min_cost_str:
            RN.min_cost[int(node)] = loaded_self_min_cost_str[node]

        # set up input variables
        cost_from_start_IN_str = loadfn(os.path.join(test_dir,
                                                     "find_path_cost_cost_from_start_OUT.json"))
        min_cost_IN_str = loadfn(os.path.join(test_dir,
                                              "find_path_cost_min_cost_OUT.json"))
        PRs_IN_str = loadfn(os.path.join(test_dir,
                                         "find_path_cost_PRs_OUT.json"))
        solved_PRs = loadfn(os.path.join(test_dir,
                                         "find_path_cost_old_solved_PRs_IN.json"))

        PRs = {}
        for node in PRs_IN_str:
            PRs[int(node)] = {}
            for start in PRs_IN_str[node]:
                PRs[int(node)][int(start)] = copy.deepcopy(PRs_IN_str[node][start])
        cost_from_start = {}
        for node in cost_from_start_IN_str:
            cost_from_start[int(node)] = {}
            for start in cost_from_start_IN_str[node]:
                cost_from_start[int(node)][int(start)] = cost_from_start_IN_str[node][start]
        min_cost = {}
        for node in min_cost_IN_str:
            min_cost[int(node)] = min_cost_IN_str[node]

        # perform calc
        solved_PRs_cal, new_solved_PRs_cal, cost_from_start_cal = RN.identify_solved_PRs(PRs, solved_PRs,
                                                                                         cost_from_start)

        # assert
        self.assertEqual(len(solved_PRs_cal), 34)
        self.assertEqual(list(set(solved_PRs_cal)-set(new_solved_PRs_cal)), [456, 556])
        self.assertEqual(len(cost_from_start_cal), 568)
        self.assertEqual(cost_from_start_cal[456][556], "no_path")
        self.assertEqual(cost_from_start_cal[556][556], 0.0)
        self.assertEqual(cost_from_start_cal[2][556], 1.6911618579132313)
        self.assertEqual(cost_from_start_cal[7][456], 1.0022887913156873)
        self.assertEqual(cost_from_start_cal[30][556], "no_path")

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_update_edge_weights(self):

        # set up RN
        RN = self.RN_cls
        RN.weight = "softplus"
        RN.graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir,
                                                                  "update_edge_weights_self_graph_IN.json")))

        # set up input variables
        min_cost_str = loadfn(os.path.join(test_dir,
                                           "update_edge_weights_min_cost_IN.json"))
        orig_graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir,
                                                                    "update_edge_weights_orig_graph_IN.json")))
        min_cost = {}
        for key in min_cost_str:
            min_cost[int(key)] = min_cost_str[key]

        # perform calc
        attrs_cal = RN.update_edge_weights(min_cost, orig_graph)

        # assert
        self.assertEqual(len(attrs_cal), 6143)
        self.assertEqual(attrs_cal[(556, '556+PR_456,421')]['softplus'], 0.2436101275766031)
        self.assertEqual(attrs_cal[(41, '41+PR_556,42')]['softplus'], 0.2606224897665045)
        self.assertEqual(attrs_cal[(308, '308+PR_556,277')]['softplus'], 0.0866554990833896)

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_final_PR_check(self):

        # set up RN
        RN = self.RN_cls
        RN.weight = "softplus"
        loaded_PRs = loadfn(os.path.join(test_dir, "finalPRcheck_PRs_HP_IN.json"))
        loaded_self_min_cost_str = loadfn(os.path.join(test_dir, "finalPRcheck_self_min_cost.json"))
        RN.graph = json_graph.adjacency_graph(loadfn(os.path.join(test_dir, "finalPRcheck_self_graph.json")))
        RN.solved_PRs = loadfn(os.path.join(test_dir, "solved_PRs_list.json"))
        loaded_PR_byproducts = loadfn(os.path.join(test_dir, "PR_byproducts_dict.json"))

        RN.PR_byproducts = {}
        for node in loaded_PR_byproducts:
            RN.PR_byproducts[int(node)] = loaded_PR_byproducts[node]

        RN.min_cost = {}
        for node in loaded_self_min_cost_str:
            RN.min_cost[int(node)] = loaded_self_min_cost_str[node]

        # set up input variables
        PRs = {}
        for node in loaded_PRs:
            PRs[int(node)] = {}
            for start in loaded_PRs[node]:
                PRs[int(node)][int(start)] = loaded_PRs[node][start]

        # perform calc
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        RN.final_PR_check(PRs)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout

        # assert
        self.assertTrue(output.__contains__("No path found from any start to PR 30"))
        self.assertTrue(output.__contains__("WARNING: Matching prereq and byproduct found! 46"))
        self.assertTrue(output.__contains__("No path found from any start to PR 513"))
        self.assertTrue(output.__contains__("No path found from any start to PR 539"))

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_or_remove_bad_nodes(self):

        # set up RN
        RN = self.RN_cls

        # set up input variables
        LEDC_ind = None
        LiEC_ind = None
        EC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break

        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break

        for entry in RN.entries["C3 H4 Li1 O3"][12][1]:
            if self.LiEC_mg.isomorphic_to(entry.mol_graph):
                LiEC_ind = entry.parameters["ind"]
                break

        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]

        nodes = [LEDC_ind, LiEC_ind, Li1_ind, EC_ind]

        # perform calc & assert
        bad_nodes_list = RN.find_or_remove_bad_nodes(nodes, remove_nodes=False)
        self.assertEqual(len(bad_nodes_list), 231)
        self.assertTrue({'511,108+112', '46+PR_556,34', '556+PR_199,192',
                         '456,399+543', '456,455'} <= set(bad_nodes_list))

        bad_nodes_pruned_graph = RN.find_or_remove_bad_nodes(nodes, remove_nodes=True)
        self.assertEqual(len(bad_nodes_pruned_graph.nodes), 10254)
        self.assertEqual(len(bad_nodes_pruned_graph.edges), 22424)
        for node_ind in nodes:
            self.assertEqual(bad_nodes_pruned_graph[node_ind], {})

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_valid_shortest_simple_paths(self):

        RN = self.RN_cls

        RN.weight = "softplus"
        loaded_graph = loadfn(os.path.join(test_dir, "graph.json"))
        RN.graph = json_graph.adjacency_graph(loaded_graph)

        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break

        paths = RN.valid_shortest_simple_paths(EC_ind, LEDC_ind)
        p = [[456, '456+PR_556,424', 424, '424,423', 423, '423,420', 420, '420+PR_41,511', 511],
            [456, '456+PR_556,424', 424, '424,423', 423, '423,420', 420, '420,41+164', 41, '41+PR_420,511', 511],
            [456, '456,455', 455, '455,448', 448, '448,51+164', 51, '51+PR_556,41', 41, '41+PR_420,511', 511],
            [456, '456+PR_556,421', 421, '421,424', 424, '424,423', 423, '423,420', 420, '420+PR_41,511', 511],
            [456, '456+PR_556,421', 421, '421,424', 424, '424,423', 423, '423,420', 420,
             '420,41+164', 41, '41+PR_420,511', 511],
            [456, '456,455', 455, '455,448', 448, '448+PR_556,420', 420, '420,41+164', 41, '41+PR_420,511', 511],
            [456, '456,455', 455, '455,448', 448, '448+PR_556,420', 420, '420+PR_41,511', 511],
            [456, '456,455', 455, '455+PR_556,423', 423, '423,420', 420, '420+PR_41,511', 511],
            [456, '456,455', 455, '455+PR_556,423', 423, '423,420', 420, '420,41+164', 41, '41+PR_420,511', 511],
            [456, '456+PR_556,424', 424, '424,423', 423, '423,420', 420, '420,419', 419, '419+PR_41,510', 510,
             '510,511', 511]]

        ind = 0
        for path in paths:
            if ind == 10:
                break
            else:
                print(path)
                self.assertEqual(path, p[ind])
                ind += 1

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_find_paths(self):

        # set up RN
        RN = ReactionNetwork.from_input_entries(self.LiEC_reextended_entries)
        RN.build()
        RN.weight = "softplus"
        # set up input variables
        EC_ind = None
        LEDC_ind = None

        for entry in RN.entries["C3 H4 O3"][10][0]:
            if self.EC_mg.isomorphic_to(entry.mol_graph):
                EC_ind = entry.parameters["ind"]
                break
        for entry in RN.entries["C4 H4 Li2 O6"][17][0]:
            if self.LEDC_mg.isomorphic_to(entry.mol_graph):
                LEDC_ind = entry.parameters["ind"]
                break
        Li1_ind = RN.entries["Li1"][0][1][0].parameters["ind"]
        print(EC_ind, Li1_ind, LEDC_ind)

        PR_paths_calculated, paths_calculated, top_paths_list = RN.find_paths([EC_ind, Li1_ind], LEDC_ind,
                                                              weight="softplus", num_paths=10)

        print(paths_calculated)

        if 420 in paths_calculated[0]["all_prereqs"]:
            self.assertEqual(paths_calculated[0]["byproducts"], [164])
        elif 41 in paths_calculated[0]["all_prereqs"]:
            self.assertEqual(paths_calculated[0]["byproducts"], [164, 164])

        self.assertAlmostEqual(paths_calculated[0]["cost"], 2.3135953094636403, 5)
        self.assertAlmostEqual(paths_calculated[0]["overall_free_energy_change"], -6.2399175587598394, 5)
        self.assertAlmostEqual(paths_calculated[0]["hardest_step_deltaG"], 0.37075842588456, 5)

        for path in paths_calculated:
            self.assertTrue(abs(path["cost"] - path["pure_cost"]) < 0.000000001)

    def test_mols_w_cuttoff(self):

        with open(os.path.join(test_dir,'RN_unittest.pkl'), 'rb') as handle:
            RN_loaded = pickle.load(handle)

        mols_to_keep, pruned_entries_list = ReactionNetwork.mols_w_cuttoff(RN_loaded, 0, build_pruned_network=False)

        self.assertEqual(len(mols_to_keep), 196)

    def test_identify_concerted_rxns_via_intermediates(self):

        with open(os.path.join(test_dir,'RN_unittest.pkl'), 'rb') as handle:
            RN_loaded = pickle.load(handle)

        with open(os.path.join(test_dir,'RN_unittest_pruned_mols_to_keep.json'), 'rb') as handle:
            mols_to_keep = pickle.load(handle)

        reactions = ReactionNetwork.identify_concerted_rxns_via_intermediates(RN_loaded, mols_to_keep,
                                                                  single_elem_interm_ignore=["C1", "H1", "O1", "Li1"])

        self.assertEqual(len(reactions), 2410)

    def test_add_concerted_rxns(self):
        with open(os.path.join(test_dir,'RN_unittest.pkl'), 'rb') as handle:
            RN_loaded = pickle.load(handle)

        with open(os.path.join(test_dir,'RN_unittest_reactions_list.json'), 'rb') as handle:
            reactions = pickle.load(handle)

        RN_loaded.add_concerted_rxns(RN_loaded, RN_loaded, reactions)

        self.assertEqual(len(RN_loaded.graph.nodes), 15064)
        self.assertEqual(len(RN_loaded.graph.edges), 36589)

if __name__ == "__main__":
    unittest.main()
