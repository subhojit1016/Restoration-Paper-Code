import random

import networkx as nx
import matplotlib.pyplot as plt

# Nodes to duplicate (input from you)
nodes_to_duplicate = ['10', '20', '30', '40','50', '60', '70', '80', '90', '100', '110', '120']

# Read the original file
with open("Network125.txt", "r") as f:
    lines = f.readlines()

# Extract existing node IDs
existing_nodes = [line.strip().split()[0] for line in lines]
existing_node_set = set(existing_nodes)

# Build new file content with duplicated lines
new_lines = []
for line in lines:
    parts = line.strip().split()
    node_id = parts[0]
    new_lines.append(line)

    if node_id in nodes_to_duplicate:
        # Get all valid predecessors (less than current node number)
        valid_preds = [n for n in existing_node_set if n.isdigit() and int(n) < int(node_id)]
        if valid_preds:
            new_pred = random.choice(valid_preds)
            # Create a duplicate line with the new predecessor
            new_line = f"{node_id}\t{new_pred}\t{parts[2]}\t{parts[3]}\t{parts[4]}\n"
            new_lines.append(new_line)

# Save the modified output
with open("Network125_Cycle.txt", "w") as f:
    f.writelines(new_lines)



# Step 1: Read the data and draw the plots

# Step 1: Read the data
# Re-import necessary packages after kernel reset

# Reload data from uploaded file
file_path = "Network125_cycle.txt"

nodes, preds, xcords, ycords, weights = [], [], [], [], []

with open(file_path, 'r') as file:
    for line in file:
        values = line.strip().split()
        nodes.append(int(values[0]))
        preds.append(int(values[1]))
        xcords.append(float(values[2]))
        ycords.append(float(values[3]))
        weights.append(int(values[4]))

# Construct the directed graph
G_child = nx.DiGraph()
edges = list(zip(preds, nodes))
G_child.add_edges_from(edges)

# Remove node 0 (depot)
if 0 in G_child:
    G_child.remove_node(0)

# Assign coordinates
all_nodes = sorted(set(nodes + preds))
node_positions = {}
for node in all_nodes:
    if node in nodes:
        idx = nodes.index(node)
        node_positions[node] = (xcords[idx], ycords[idx])
    else:
        node_positions[node] = (sum(xcords) / len(xcords), sum(ycords) / len(ycords))

# Plotting
fig, ax = plt.subplots(figsize=(16, 12))
edges_list = list(G_child.edges())

nx.draw_networkx_nodes(G_child, node_positions, node_size=350, node_color='white', ax=ax)
nx.draw_networkx_labels(G_child, node_positions, font_size=22)
nx.draw_networkx_edges(G_child, node_positions, edgelist=edges_list, arrows=True)

ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=26)
plt.xlabel("X-axis (Width of the region)", fontsize=26)
plt.ylabel("Y-axis (Height of the region)", fontsize=26)
plt.title("Power Network", fontsize=26)
plt.savefig("Output_plot.png", dpi=600)
plt.grid(True)
plt.tight_layout()
plt.show()
