from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import pulp
from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, lpSum

from mrnet.core.mol_entry import MoleculeEntry

# typing
Bond = Tuple[int, int]
AtomMappingDict = Dict[int, int]


def get_reaction_atom_mapping(
    reactants: List[MoleculeEntry],
    products: List[MoleculeEntry],
    max_bond_change: int = 10,
) -> Tuple[List[AtomMappingDict], List[AtomMappingDict], int]:
    """
    Get the atom mapping between the reactants and products of a reaction.

    This works for any number of reactant/product molecules, but the reaction should be
    stoichiometrically balanced. This implementation only respects atom type and ignore
    bond type (e.g. single vs double) as well as other information like stereo centers.

    There could be multiple mappings (due to, e.g. isomorphism and the fact bond type is
    not taken into consideration), and this function only returns one such mapping.

    The algorithm treats the reactants as a single disjoint graph (same for the products)
    and using integer programming to find the smallest bond edits to transform the
    reactant graph to the product graph. See the paper in `Reference` for details of the
    algorithm.

    Args:
        reactants: reactant molecules
        products: product molecules
        max_bond_change: maximum number of allowed bond changes (break and form)
            between the reactants and products.

    Returns:
        reactants_map_number: rdkit style atom map number for the reactant molecules
            (starts from 1 in rdkit but from 0 here). Each dict holds the map number
            for one molecule {atom_index: map_number}, which maps the atom index to
            a map number. Atoms in the reactants and products having the same map number
            corresponds to each other in the reaction.
        products_map_number: rdkit style atom map number for the product molecules.
            See `reactants_map_number` for more.
        num_bond_change: number of changed bond in the reaction

    References:
        `Stereochemically Consistent Reaction Mapping and Identification of Multiple
        Reaction Mechanisms through Integer Linear Optimization`,
        J. Chem. Inf. Model. 2012, 52, 84–92, https://doi.org/10.1021/ci200351b
    """

    # preliminary check

    # check 1: reactants and products have the same atom counts
    rct_species = defaultdict(int)
    prdt_species = defaultdict(int)
    for m in reactants:
        for s in m.species:
            rct_species[s] += 1
    for m in products:
        for s in m.species:
            prdt_species[s] += 1
    if rct_species != prdt_species:
        raise ReactionMappingError(
            "Expect reactants and products to have the same atom count, "
            f"but got {dict(rct_species)} and {dict(prdt_species)}."
        )

    # check 2: number of bond change smaller than allowed maximum
    # This only checks the number of bonds and thus actual num changes could be larger,
    # which will be checked later.
    num_bond_change = abs(
        sum(len(m.bonds) for m in reactants) - sum(len(m.bonds) for m in products)
    )
    if num_bond_change > max_bond_change:
        raise ReactionMappingError(
            f"Number of changed bond is at least {num_bond_change}, larger than allowed "
            f"maximum {max_bond_change}"
        )

    # local and global atom index mapping
    (
        reactant_species,
        reactant_bonds,
        _,
        reactant_idx_mapping,
    ) = get_local_global_atom_index_mapping(reactants)
    (
        product_species,
        product_bonds,
        _,
        product_idx_mapping,
    ) = get_local_global_atom_index_mapping(products)

    # solve integer programming problem to get atom mapping
    if len(reactant_bonds) == 0 or len(product_bonds) == 0:
        num_bond_change, r2p_mapping, p2r_mapping = get_atom_mapping_no_bonds(
            reactant_species, product_species, reactant_bonds, product_bonds
        )
    else:
        num_bond_change, r2p_mapping, p2r_mapping = solve_integer_programing(
            reactant_species, product_species, reactant_bonds, product_bonds
        )

    # final check
    if num_bond_change > max_bond_change:
        raise ReactionMappingError(
            f"Number of bond change {num_bond_change} larger than allowed maximum number "
            f"of bond change {max_bond_change}."
        )

    if None in r2p_mapping:
        global_idx = r2p_mapping.index(None)
        mol_idx, atom_idx = reactant_idx_mapping[global_idx]
        raise ReactionMappingError(
            f"Cannot find mapping for atom {atom_idx} of reactant molecule {mol_idx}."
        )

    if None in p2r_mapping:
        global_idx = p2r_mapping.index(None)
        mol_idx, atom_idx = product_idx_mapping[global_idx]
        raise ReactionMappingError(
            f"Cannot find mapping for atom {atom_idx} of product molecule {mol_idx}."
        )

    # Everything is alright, create atom map number.
    # Atoms in reactants will have their global index as map number.
    # Map number for atoms in products are determined accordingly based on the results
    # of integer programming
    reactants_map_number = [{} for _ in range(len(reactants))]
    products_map_number = [{} for _ in range(len(products))]
    for rct_idx, prdt_idx in enumerate(r2p_mapping):
        map_number = rct_idx

        mol_idx, atom_idx = reactant_idx_mapping[rct_idx]
        reactants_map_number[mol_idx][atom_idx] = map_number

        mol_idx, atom_idx = product_idx_mapping[prdt_idx]
        products_map_number[mol_idx][atom_idx] = map_number

    return reactants_map_number, products_map_number, num_bond_change


def get_local_global_atom_index_mapping(
    molecules: List[MoleculeEntry],
) -> Tuple[List[str], List[Bond], List[List[int]], List[Tuple[int, int]]]:
    """
    Map the local and global indices of atoms in a sequence of mols.

    This is a utility function for `get_reaction_atom_mapping()`.

    Think of this as combining a sequence of molecules into a single molecule and then
    relabelling the atom index in each mol to form a consecutive global index in the
    combined molecule.

    Local indices for atoms in each mol are [0, ..., N-1], where N is the number of
    atoms in the corresponding atoms.

    Global indices for atoms in the 1st mol is [0, ..., N1-1],
    in the 2nd mol is [N1, ..., N1+N2-1],
    in the 3rd mol is [N1+N2, ..., N1+N2+N3-1]
    ...
    where N1, N2, and N3 are the number of atoms in molecules 1, 2, and 3.

    Args:
        molecules: A sequence of molecule entry.

    Returns:
        global_species: species of atoms in the combined molecule.
        global_bonds: all bonds in the combine molecule; each bond is indexed by a
            tuple of global atom index.
        local_to_global: local atom index to global atom index. Each inner list holds
            the global atom indexes of a molecule. E.g. local_to_global[0][2] gives 4,
            meaning atom 2 of molecule 0 has a global index of 4.
        global_to_local: global atom index to local atom index. Each tuple
            (mol_index, atom_index) is for one atom, with mol_index the index of the
            molecule from which the atom is from and atom_index the local index of the
            atom in the molecule. E.g. global[4] gives a tuple (0, 2), meaning atom with
            global index 4 corresponds to atom 2 in molecule 0.
    """

    global_species = []
    global_bonds = []

    local_to_global = []
    global_to_local = []

    n = 0
    for i, m in enumerate(molecules):
        global_species.extend(m.species)

        bonds = np.asarray(m.bonds) + n
        global_bonds.extend([tuple(b) for b in bonds.tolist()])

        mp = [j + n for j in range(m.num_atoms)]
        local_to_global.append(mp)

        mp = [(i, j) for j in range(m.num_atoms)]
        global_to_local.extend(mp)

        n += m.num_atoms

    return global_species, global_bonds, local_to_global, global_to_local


def solve_integer_programing(
    reactant_species: List[str],
    product_species: List[str],
    reactant_bonds: List[Bond],
    product_bonds: List[Bond],
) -> Tuple[int, List[Union[int, None]], List[Union[int, None]]]:
    """
    Solve an integer programming problem to get atom mapping between reactants and
    products.

    This is a utility function for `get_reaction_atom_mapping()`.

    Args:
        reactant_species: species string of reactant atoms
        product_species: species string of product atoms
        reactant_bonds: bonds in reactant
        product_bonds: bonds in product

    Returns:
        objective: minimized objective value. This corresponds to the number of changed
            bonds (both broken and formed) in the reaction.
        r2p_mapping: mapping of reactant atom to product atom, e.g. r2p_mapping[0]
            giving 3 means that reactant atom 0 maps to product atom 3. A value of
            `None` means a mapping cannot be found for the atom.
        p2r_mapping: mapping of product atom to reactant atom. See `r2p_mapping` for an
            example.

    Reference:
        `Stereochemically Consistent Reaction Mapping and Identification of Multiple
        Reaction Mechanisms through Integer Linear Optimization`,
        J. Chem. Inf. Model. 2012, 52, 84–92, https://doi.org/10.1021/ci200351b
    """

    atoms = list(range(len(reactant_species)))

    # init model and variables
    model = LpProblem(sense=LpMinimize)

    y_vars = {
        (i, k): LpVariable(cat=LpBinary, name=f"y_{i}_{k}")
        for i in atoms
        for k in atoms
    }

    alpha_vars = {
        (i, j, k, l): LpVariable(cat=LpBinary, name=f"alpha_{i}_{j}_{k}_{l}")
        for (i, j) in reactant_bonds
        for (k, l) in product_bonds
    }

    # add constraints

    # constraint 2: each atom in the reactants maps to exactly one atom in the products
    # constraint 3: each atom in the products maps to exactly one atom in the reactants
    for i in atoms:
        model += lpSum([y_vars[(i, k)] for k in atoms]) == 1
    for k in atoms:
        model += lpSum([y_vars[(i, k)] for i in atoms]) == 1

    # constraint 4: allows only atoms of the same type to map to one another
    for i in atoms:
        for k in atoms:
            if reactant_species[i] != product_species[k]:
                model += y_vars[(i, k)] == 0

    # constraints 5 and 6: define each alpha_ijkl variable, permitting it to take the
    # value of one only if the reactant bond (i,j) maps to the product bond (k,l)
    for (i, j) in reactant_bonds:
        for (k, l) in product_bonds:
            model += alpha_vars[(i, j, k, l)] <= y_vars[(i, k)] + y_vars[(i, l)]
            model += alpha_vars[(i, j, k, l)] <= y_vars[(j, k)] + y_vars[(j, l)]

    # create objective
    obj1 = lpSum(
        1 - lpSum(alpha_vars[(i, j, k, l)] for (k, l) in product_bonds)
        for (i, j) in reactant_bonds
    )
    obj2 = lpSum(
        1 - lpSum(alpha_vars[(i, j, k, l)] for (i, j) in reactant_bonds)
        for (k, l) in product_bonds
    )
    objective = obj1 + obj2

    # solve the problem
    try:
        model.setObjective(objective)
        model.solve()
    except Exception:
        raise ReactionMappingError("Failed solving integer programming.")

    objective = pulp.value(model.objective)
    if objective is None:
        raise ReactionMappingError("Failed solving integer programming.")

    # get atom mapping between reactant and product
    r2p_mapping = [None for _ in atoms]
    p2r_mapping = [None for _ in atoms]
    for (i, k), v in y_vars.items():
        if pulp.value(v) == 1:
            r2p_mapping[i] = k
            p2r_mapping[k] = i

    return objective, r2p_mapping, p2r_mapping


def get_atom_mapping_no_bonds(
    reactant_species: List[str],
    product_species: List[str],
    reactant_bonds: List[Bond],
    product_bonds: List[Bond],
) -> Tuple[int, List[Union[int, None]], List[Union[int, None]]]:
    """
    Get the atom mapping for reaction where there is no bonds in either the reactants
    or products. For example, a reaction C-O -> C + O.


    This is a complement function to `solve_integer_programing()`, which cannot deal
    with the case where there is no bonds in the reactants or products.

    See `solve_integer_programing()` for the arguments and returns.
    """

    if len(reactant_bonds) != 0 and len(product_bonds) != 0:
        raise ReactionMappingError(
            "Expect either reactants or products has 0 bonds, but reactants has "
            f"{len(reactant_bonds)} and products has {len(product_bonds)}."
        )

    # the only thing we need to do is to match species
    product_species_to_index = defaultdict(list)
    for i, s in enumerate(product_species):
        product_species_to_index[s].append(i)

    r2p_mapping = []
    for s in reactant_species:
        r2p_mapping.append(product_species_to_index[s].pop())
    p2r_mapping = [r2p_mapping.index(i) for i in range(len(product_species))]

    # objective, i.e. number of bond change
    objective = abs(len(reactant_bonds) - len(product_bonds))

    return objective, r2p_mapping, p2r_mapping


class ReactionMappingError(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)
        self.msg = msg
