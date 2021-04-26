import os
import numpy as np
from monty.serialization import dumpfn, loadfn
from mrnet.network.reaction_network import ReactionNetwork
from mrnet.core.mol_entry import MoleculeEntry
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.fragmenter import metal_edge_extender
import copy
import pickle

__author__ = "Aniruddh Khanwale"
__email__ = "akhanwale@lbl.gov"
__copyright__ = "Copyright 2021, The Materials Project"
__version__ = "0.1"

test_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "test_files",
    "reaction_network_files",
)

try:
    import openbabel as ob
except ImportError:
    ob = None
""" 
Create a reaction network for testing from LiEC entries
"""
if ob:
    EC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(test_dir, "EC.xyz")), OpenBabelNN()
    )
    EC_mg = metal_edge_extender(EC_mg)

    LiEC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(test_dir, "LiEC.xyz")), OpenBabelNN()
    )
    LiEC_mg = metal_edge_extender(LiEC_mg)
    LEDC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(test_dir, "LEDC.xyz")), OpenBabelNN()
    )
    LEDC_mg = metal_edge_extender(LEDC_mg)
    LEMC_mg = MoleculeGraph.with_local_env_strategy(
        Molecule.from_file(os.path.join(test_dir, "LEMC.xyz")), OpenBabelNN()
    )
    LEMC_mg = metal_edge_extender(LEMC_mg)
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
reaction_network = ReactionNetwork.from_input_entries(
    LiEC_reextended_entries, electron_free_energy=-2.15
)
reaction_network.build()
for entry in reaction_network.entries["C3 H4 Li1 O3"][12][1]:
    if LiEC_mg.isomorphic_to(entry.mol_graph):
        LiEC_ind = entry.parameters["ind"]
        break
for entry in reaction_network.entries["C3 H4 O3"][10][0]:
    if EC_mg.isomorphic_to(entry.mol_graph):
        EC_ind = entry.parameters["ind"]
        break
for entry in reaction_network.entries["C4 H4 Li2 O6"][17][0]:
    if LEDC_mg.isomorphic_to(entry.mol_graph):
        LEDC_ind = entry.parameters["ind"]
        break
Li1_ind = reaction_network.entries["Li1"][0][1][0].parameters["ind"]
pickle_in = open(os.path.join(test_dir, "unittest_RN_build.pkl"), "wb")
pickle.dump(reaction_network, pickle_in)
pickle_in.close()
reaction_network.solve_prerequisites(
    [EC_ind, Li1_ind], weight="softplus", generate_test_files=True
)
pickle_in = open(os.path.join(test_dir, "unittest_RN_pr_solved.pkl"), "wb")
pickle.dump(reaction_network, pickle_in)
pickle_in.close()
pickle_in = open(os.path.join(test_dir, "unittest_RN_pr_solved_PRs.pkl"), "wb")
pickle.dump(reaction_network.PRs, pickle_in)
pickle_in.close()
