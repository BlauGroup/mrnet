from typing import List, Union, Optional, Tuple
from mrnet.entries import BasicEntry
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from dataclasses import dataclass


@dataclass
class MoleculeEntry(BasicEntry):
    """
    An entry with molecular graph structure
    """

    molecule_graph: MoleculeGraph

    @property
    def molecule(self) -> Molecule:
        return self.molecule_graph.molecule


    @property
    def graph(self) -> nx.MultiDiGraph:
        return self.mol_graph.graph


    @property
    def charge(self) -> float:
        return self.molecule.charge

    @property
    def formula(self) -> str:
        return self.molecule.composition.alphabetical_formula
    
    property
    def species(self) -> List[str]:
        return [str(s) for s in self.molecule.species]

    @property
    def bonds(self) -> List[Tuple[int, int]]:
        return [(int(sorted(e)[0]), int(sorted(e)[1])) for e in self.graph.edges()]

    @property
    def num_atoms(self) -> int:
        return len(self.molecule)

    @property
    def num_bonds(self) -> int:
        return len(self.bonds)

    @property
    def coords(self) -> np.ndarray:
        return self.molecule.cart_coords


    def get_fragments(self) -> Dict[Tuple[int, int], List[MoleculeGraph]]]:
        """
        Get the fragments of the molecule by breaking all its bonds.

        Returns:
            Fragments dict {(atom1, atom2): [fragments]}, where
                the key `(atom1, atom2)` specifies the broken bond indexed by the two
                atoms forming the bond, and the value `[fragments]` is a list of
                fragments obtained by breaking the bond. This list can have either one
                element (ring-opening A->B) or two elements (not ring-opening A->B+C).
                The dictionary is empty if the molecule has no bonds (e.g. Li+).
        """

        fragments = {}

        for edge in self.bonds:
            try:
                frags = self.mol_graph.split_molecule_subgraphs(
                    [edge], allow_reverse=True, alterations=None
                )
                fragments[edge] = frags

            except MolGraphSplitError:
                # cannot split (ring-opening editing)
                frag = copy.deepcopy(self.mol_graph)
                idx1, idx2 = edge
                frag.break_edge(idx1, idx2, allow_reverse=True)
                fragments[edge] = [frag]

        return fragments










