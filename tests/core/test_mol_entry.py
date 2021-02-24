import os
import pytest
import numpy as np
from monty.serialization import loadfn
from pymatgen import Molecule
from mrnet.core.mol_entry import MoleculeEntry

try:
    import openbabel as ob
except ImportError:
    ob = None

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test_files")


def make_a_mol_entry():
    r"""
    Make a symmetric (fake) molecule with ring.
                O(0)
               / \
              /   \
      H(1)--C(2)--C(3)--H(4)
             |     |
            H(5)  H(6)
    """
    species = ["O", "H", "C", "C", "H", "H", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [-0.5, -1.0, 0.0],
        [0.5, -1.0, 0.0],
    ]

    m = Molecule(species, coords)
    entry = MoleculeEntry(m, energy=0.0)

    return entry


class TestMolEntry:
    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_from_libe():
        doc = loadfn(os.path.join(test_dir, "libe_entry.json"))
        entry = MoleculeEntry.from_dataset_entry(doc)

        assert entry.entry_id == "libe-120825"
        assert entry.get_free_energy() == -11366.853316264207

        entry_rrho = MoleculeEntry.from_dataset_entry(doc, use_thermo="rrho_shifted")
        assert entry_rrho.get_free_energy() == -11366.84673089201

        entry_qrrho = MoleculeEntry.from_dataset_entry(doc, use_thermo="qrrho")
        assert entry_qrrho.get_free_energy() == -11366.846521648069

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_property():
        mol_doc = loadfn(os.path.join(test_dir, "mol_doc_C1H1O2.json"))
        entry = MoleculeEntry.from_molecule_document(mol_doc)

        assert entry.entry_id == 215871
        assert entry.formula == "C1 H1 O2"
        assert entry.charge == -1

        assert entry.energy == -189.256491214986
        assert entry.entropy == 59.192
        assert entry.enthalpy == 14.812
        assert entry.get_free_energy() == -5150.055177181075

        assert entry.species == ["O", "C", "O", "H"]
        assert entry.num_atoms == 4
        assert entry.bonds == [(0, 1), (0, 3), (1, 2)]
        assert entry.num_bonds == 3

        ref_coords = [
            [3.6564233496, -2.7122919826, 0.0],
            [3.2388110663, -1.3557709432, 0.0],
            [4.1820830042, -0.5664108935, 0.0],
            [4.6310495799, -2.6817671807, 0.0],
        ]
        assert np.array_equal(entry.coords, ref_coords)

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_get_fragments():
        entry = make_a_mol_entry()
        fragments = entry.get_fragments()

        assert len(fragments) == 7

        # break bond yields 1 fragment
        assert len(fragments[(0, 2)]) == 1
        assert len(fragments[(0, 3)]) == 1
        assert len(fragments[(2, 3)]) == 1

        # break bond yields 2 fragments
        assert len(fragments[(1, 2)]) == 2
        assert len(fragments[(2, 5)]) == 2
        assert len(fragments[(3, 4)]) == 2
        assert len(fragments[(3, 6)]) == 2

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_get_isomorphic_bonds():
        entry = make_a_mol_entry()
        iso_bonds = entry.get_isomorphic_bonds()

        # sort iso_bonds for easier comparison
        iso_bonds = sorted([sorted(group) for group in iso_bonds])

        assert iso_bonds == [
            [(0, 2), (0, 3)],
            [(1, 2), (2, 5), (3, 4), (3, 6)],
            [(2, 3)],
        ]
