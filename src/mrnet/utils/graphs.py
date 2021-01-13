from typing import List, Tuple, Set

from pymatgen.analysis.graphs import MoleculeGraph


def extract_bond_environment(
    mg: MoleculeGraph, bonds: List[Tuple[int, int]], order=1
) -> set:
    """
    Extract the local environment of a particular chemical bond in a MoleculeGraph

    :param bonds:
    :param order:

    :return: set of integers representing the relevant atom indices
    """

    indices = set()  # type: Set[int]
    if order < 0:
        return indices
    elif order == 0:
        for bond in bonds:
            indices.add(bond[0])
            indices.add(bond[1])
        return indices
    else:
        graph = mg.graph.to_undirected()
        for bond in bonds:
            sub_bonds = list()
            for neighbor in graph[bond[0]]:
                sub_bonds.append((bond[0], neighbor))
            for neighbor in graph[bond[1]]:
                sub_bonds.append((bond[1], neighbor))
            indices = indices.union(extract_bond_environment(mg, sub_bonds, order - 1))
        return indices
