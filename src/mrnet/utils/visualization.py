import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

    atom_colors = {"O": "red", "H": "gray", "C": "black", "Li": "purple",
                   "N": "blue", "S": "yellow", "Cl": "green", "P": "orange",
                   "F": "#03C04A", "B": "#FFAA77", "Mg": "#FF66CC"}

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
