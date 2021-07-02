import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import TextIO, Dict
from mrnet.core.mol_entry import MoleculeEntry
import os

from copy import deepcopy


def visualize_molecule_count_histogram(final_counts, path):
    """
    generates a histogram for a list of integers
    """
    fig = plt.figure()
    ax = plt.axes()
    bins = np.arange(0, max(final_counts) + 1.5) - 0.5
    ax.set_xticks(bins + 0.5)
    ax.hist(final_counts, bins)
    fig.savefig(path)


def visualize_molecule_entry(molecule_entry, path):
    """
    visualize a molecule using graphviz and
    output the resulting pdf to path
    """

    atom_colors = {
        "O": "red",
        "H": "gray",
        "C": "black",
        "Li": "purple",
        "F": "green",
        "P": "orange",
    }

    graph = deepcopy(molecule_entry.graph).to_undirected()

    nx.set_node_attributes(graph, "", "label")
    nx.set_node_attributes(graph, "filled", "style")
    nx.set_node_attributes(graph, "circle", "shape")
    nx.set_node_attributes(graph, "0.2", "width")
    nx.set_node_attributes(
        graph,
        dict(enumerate([atom_colors[a] for a in molecule_entry.species])),
        "color",
    )

    charge = molecule_entry.charge
    agraph = nx.nx_agraph.to_agraph(graph)
    if charge != 0:
        agraph.add_node(
            "charge",
            label=str(charge),
            fontsize="25.0",
            shape="box",
            color="gray",
            style="dashed, rounded",
        )

    agraph.layout()
    agraph.draw(path, format="pdf")


def visualize_molecules(folder: str, mol_entries: Dict[int, MoleculeEntry]):
    if os.path.isdir(folder):
        return

    os.mkdir(folder)
    for index, molecule_entry in mol_entries.items():
        visualize_molecule_entry(molecule_entry, folder + "/" + str(index) + ".pdf")


def generate_latex_header(f: TextIO):
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{graphicx}\n")
    f.write("\\usepackage[margin=1cm]{geometry}\n")
    f.write("\\usepackage{amsmath}\n")
    f.write("\\pagenumbering{gobble}\n")
    f.write("\\begin{document}\n")


def generate_latex_footer(f: TextIO):
    f.write("\\end{document}")


def latex_emit_molecule(f: TextIO, species_index: int):
    """
    in the same folder as the latex doc being produced, there
    should be a folder molecule_diagrams which contains the pdfs
    for all the molecules labeled as species_index.pdf.
    """
    f.write(str(species_index) + "\n")
    f.write(
        "\\raisebox{-.5\\height}{"
        + "\\includegraphics[scale=0.2]{./molecule_diagrams/"
        + str(species_index)
        + ".pdf}}\n"
    )


def latex_emit_reaction(f: TextIO, reaction: dict, reaction_index=None):
    """
    reaction should be a dictionary with keys
    "reactants"
    "products"
    "dG"
    """
    f.write("$$\n")
    first = True
    if reaction_index is not None:
        f.write(str(reaction_index) + ":\n")
    for reactant_index in reaction["reactants"]:
        if first:
            first = False
        else:
            f.write("+\n")

        latex_emit_molecule(f, reactant_index)

    f.write("\\xrightarrow{" + ("%.2f" % reaction["dG"]) + "}\n")

    first = True
    for product_index in reaction["products"]:
        if first:
            first = False
        else:
            f.write("+\n")

        latex_emit_molecule(f, product_index)

    f.write("$$")
    f.write("\n\n\n")
