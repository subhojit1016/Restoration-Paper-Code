import matplotlib.pyplot as plt
import networkx as nx

# Node positions
positions = {
    1: (2, 3),
    2: (1.5, 2),
    3: (2.8, 2.2),   # moved slightly right and up
    4: (1, 1),
    5: (2.2, 1.4),   # moved slightly left
    6: (2.8, 0.9)    # moved further right/down
}

# Power network edges (directed)
edges = [(1, 2), (1, 3), (2, 4), (2, 5), (5, 6)]

# Colors
GRAY = {"white": "#FFFFFF", "light": "#BBBBBB", "verylight": "#EEEEEE", "veryverylight": "#F5F5F5"}

color_before = {1: GRAY["white"], 2: GRAY["light"], 3: GRAY["white"], 4: GRAY["verylight"], 5: GRAY["verylight"], 6: GRAY["verylight"]}
shapes_before = {'o': [1, 2, 3], 's': [4, 5, 6]}

def plot_network(edges, node_colors, node_shapes, title):
    fig, ax = plt.subplots(figsize=(6, 5))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, positions, edgelist=edges,
        ax=ax, edge_color='black',
        arrows=True, arrowstyle='-|>', arrowsize=30,
        width=2, connectionstyle="arc3,rad=0",
        min_source_margin=15, min_target_margin=15
    )

    # Draw nodes by shape
    for shape, shape_nodes in node_shapes.items():
        nx.draw_networkx_nodes(
            G, positions, nodelist=shape_nodes,
            node_shape=shape,
            node_color=[node_colors[n] for n in shape_nodes],
            edgecolors='black',
            node_size=1500, ax=ax
        )

    # Node labels
    nx.draw_networkx_labels(G, positions, ax=ax, font_size=12, font_weight='bold')

    # ---- Axis setup & FORCE ticks to show ----
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.grid(True, linestyle="--", color="black", linewidth=0.8, alpha=0.8)

    ax.set_xlim(0.5, 3.0)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3.0])
    ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3.0, 3.5])

    # Force axis/ticks/spines visible (NetworkX can hide them)
    ax.set_axis_on()
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='both',
                   bottom=True, left=True, top=False, right=False,
                   labelbottom=True, labelleft=True, labelsize=10, colors='black')

    plt.tight_layout()
    plt.show()

# Example
plot_network(edges, color_before, shapes_before, "")
