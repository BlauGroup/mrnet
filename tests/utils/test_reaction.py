from pathlib import Path

from monty.serialization import loadfn

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.utils.reaction import (
    get_atom_mapping_no_bonds,
    get_local_global_atom_index_mapping,
    get_reaction_atom_mapping,
    solve_integer_programing,
)

test_dir = Path(__file__).parent.parent.parent.joinpath(
    "test_files", "utils_reaction_files"
)


def test_get_atom_mapping_no_bonds():
    reactant_species = ["C", "H", "O", "H"]
    product_species = ["H", "H", "C", "O"]
    reactant_bonds = [(0, 1), (0, 2), (0, 3)]
    product_bonds = []

    num_change_bond, r2p_mapping, p2r_mapping = get_atom_mapping_no_bonds(
        reactant_species, product_species, reactant_bonds, product_bonds
    )

    assert num_change_bond == 3
    assert r2p_mapping == [2, 1, 3, 0]
    assert p2r_mapping == [3, 1, 0, 2]


def test_get_local_global_atom_index_mapping():
    reactants, _ = _load_reaction()
    species, bonds, l2g, g2l = get_local_global_atom_index_mapping(reactants)

    ref_species = ["C", "C", "O", "C", "O", "O", "Li", "H", "H", "H", "O", "Li", "H"]
    ref_bonds = [
        (0, 1),
        (0, 6),
        (1, 7),
        (1, 8),
        (2, 6),
        (2, 3),
        (3, 5),
        (3, 4),
        (5, 9),
        (5, 6),
        (10, 11),
        (10, 12),
    ]
    ref_l2g = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12]]
    ref_g2l = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (1, 0),
        (1, 1),
        (1, 2),
    ]

    assert species == ref_species
    assert bonds == ref_bonds
    assert l2g == ref_l2g
    assert g2l == ref_g2l


def test_slove_integer_programming():
    reactants, products = _load_reaction()
    rct_species, rct_bonds, _, _ = get_local_global_atom_index_mapping(reactants)
    prdt_species, prdt_bonds, _, _ = get_local_global_atom_index_mapping(products)

    num_change_bond, r2p_mapping, p2r_mapping = solve_integer_programing(
        rct_species, prdt_species, rct_bonds, prdt_bonds
    )

    assert num_change_bond == 2
    assert r2p_mapping == [8, 7, 0, 1, 2, 3, 4, 12, 11, 5, 6, 9, 10]
    assert p2r_mapping == [2, 3, 4, 5, 6, 9, 10, 1, 0, 11, 12, 8, 7]


def test_get_reaction_atom_mapping():
    reactants, products = _load_reaction()
    rct_map_number, prdt_map_number, num_change_bond = get_reaction_atom_mapping(
        reactants, products
    )

    assert num_change_bond == 2
    assert rct_map_number == [
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        {0: 10, 1: 11, 2: 12},
    ]
    assert prdt_map_number == [
        {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 9},
        {2: 0, 1: 1, 6: 7, 5: 8, 0: 10, 3: 11, 4: 12},
    ]


def _load_reaction():
    filename = test_dir.joinpath("rxn_mol_graphs.json")
    mol_graphs = loadfn(filename)

    mol_entries = [MoleculeEntry(m.molecule, mol_graph=m) for m in mol_graphs]
    reactants = mol_entries[:2]
    products = mol_entries[2:]

    return reactants, products
