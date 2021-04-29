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

concerted_entries = loadfn(os.path.join(test_dir, "pf_unittest_entries_241.json"))
reaction_network = ReactionNetwork.from_input_entries(
    concerted_entries, electron_free_energy=-2.15
)
reaction_network.build()
pickle_in = open(
    os.path.join(test_dir, "identify_concerted_via_intermediate_unittest_RN.pkl"),
    "wb",
)
pickle.dump(reaction_network, pickle_in)
pickle_in.close()
