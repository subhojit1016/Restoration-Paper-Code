# ==========================================================================
# Generated_instance_v3.0_updated_latest.py
# Purpose: Generate network instances using the updated instance-construction routines.
# Publication note: Original executable statements, names, and parameters are unchanged.
# Inputs and outputs: Check the file paths and experiment settings below before running.
# ==========================================================================

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import itertools as it
import random as rn
import csv
import copy
from scipy.spatial import distance
from itertools import groupby
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import timeit
from scipy.spatial.distance import cityblock
import pickle
from collections import Counter
import ast
import seaborn as sns

###############   Input parameters that needs to be modified
numofnodes = 125 #Change it depending on how many nodes you want in the network
line_network = []
for i in range(1, numofnodes + 1):
    line_network.append(i)

degree_node = 6 # what should be the maximum degree of node you want

no_of_nodes_to_select = int(numofnodes/4) #Selects the number of nodes whose parent you want to change. Change the denominator accordingly


start_node = 1
new_name = "1"
count_modify = 5 # This shows how many different networks you want to have by modifying the depth


origin_x = 1
origin_y = 1

rectangle_width = 11
rectangle_height = 11

##########   Parameter update finished       ##############################
# Generate node locations inside the specified rectangular region.
def place_points_inside_rectangle(rectangle_width, rectangle_height, num_points):
    points = []
    for _ in range(num_points):
        x = rn.uniform(origin_x, origin_x + rectangle_width)
        y = rn.uniform(origin_y, origin_y + rectangle_height)
        points.append((x, y))
    return points

# Example usage

points1 = place_points_inside_rectangle(rectangle_width, rectangle_height, numofnodes)

# Print the coordinates of the points





x_cord=[]
y_cord=[]
#for k in range(len(points)):
#    x_cord.append(points[k][0])
#    y_cord.append(points[k][1])
    
for k in range(len(points1)):
    x_cord.append(points1[k][0])
    y_cord.append(points1[k][1])
    
y_cord = np.array(y_cord)
x_cord = np.array(x_cord)

sorted_indices = np.argsort(x_cord)

# Use the sorted_indices to rearrange the elements in the array Y

sorted_Y = y_cord[sorted_indices]
sorted_X = x_cord[sorted_indices]


# Scatter plot
sns.scatterplot(x=sorted_X, y=sorted_Y, s=200, c='g')
for i in range(len(line_network)):
    plt.annotate(line_network[i], (sorted_X[i], sorted_Y[i]+ 1))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title(f"Location of {numofnodes} nodes")
plt.grid(True)
#plt.title("Location of the Nodes", fontsize=15)
plt.xlim((0, rectangle_width + 5))
plt.ylim((0, rectangle_height + 5 ))
plt.xlabel("X-distance", fontsize=15)
plt.ylabel("Y-distance", fontsize=15)
plt.savefig(f"plot_{numofnodes}.png")
plt.show()



weight = np.ones(numofnodes)
df_location = pd.DataFrame({'x':line_network, 'xcord':sorted_X, 'ycord':sorted_Y, 'weight':weight})


with open(f"Location_{numofnodes}.txt", 'w') as f:
    for index, row in df_location.iterrows():
        # Convert the first two columns to integers
        integer_values = '\t'.join(str(int(value)) for value in row[:1])
        # Convert the next two columns to floats
        float_values = '\t'.join(f'{value:.3f}' for value in row[1:3])
        # Convert the last column to an integer
        last_integer_value = int(row[-1])
        # Combine all values and write to the file
        f.write(f"{integer_values}\t{float_values}\t{last_integer_value}\n")




##############Start Creating the power network topology########################


network_name=[]
depth_tree=[]
avg_outgoing_degree=[]
min_outgoing_degree=[]
max_outgoing_degree=[]


f = open(f"Location_{numofnodes}.txt",'r')

# Open the file in read mode ('r')
with f as file:
    # Read the lines of the file
    lines = file.readlines()




# Register a vertex in the network representation.
def add_vertex(v):
    global distance_matrix
    global vertices_no
    global vertices
    if v in vertices:
        print("Vertex ", v, " already exists")
    else:
        vertices_no = vertices_no + 1
    vertices.append(v)
    if vertices_no > 1:
        for vertex in distance_matrix:
            vertex.append(0)
    temp = []
    for i in range(vertices_no):
        temp.append(0)
    distance_matrix.append(temp)

# Add an edge between vertex v1 and v2 with edge weight e
# Add an edge to the network representation.
def add_edge(v1, v2, e):
    global distance_matrix
    global vertices_no
    global vertices
    # Check if vertex v1 is a valid vertex
    if v1 not in vertices:
        print("Vertex ", v1, " does not exist.")
    # Check if vertex v1 is a valid vertex
    elif v2 not in vertices:
        print("Vertex ", v2, " does not exist.")

    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        distance_matrix[index1][index2] = e


# Print the graph
# Display or report the constructed network.
def print_graph():
  global distance_matrix
  global vertices_no
  for i in range(vertices_no):
    for j in range(vertices_no):
      if distance_matrix[i][j] != 0:
        print(vertices[i], " -> ", vertices[j], \
        " edge weight: ", distance_matrix[i][j])


# Check a condition used by the algorithm.
def check(i,j):
    if (distance_matrix[i][j]!=0):
        return True
    else:
        return False

vertices = []
# stores the number of vertices in the graph
vertices_no = 0
distance_matrix = []
# Add vertices to the graph
#distance_matrix[0][14]


nodes = []
predecessor = []
xcord = []
ycord = []
weight =[]



for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    #predecessor.append(int(values[1]))
    xcord.append(float(values[1]))
    ycord.append(float(values[2]))
    weight.append(int(values[3]))

    
#nodes_updated = copy.deepcopy(nodes)
#predecessor_update = copy.deepcopy(predecessor)
#nodes_updated.remove(1)
#predecessor_update.remove(0)

#for i,j in zip (nodes_updated,predecessor_update):
#    G.add_edges_from([(j, i)])



#potential_nodes=list(G.nodes())

update_potentialnodes = copy.deepcopy(nodes)
update_potentialnodes.insert(0,0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0,np.mean(xcord))
ycord.insert(0,np.mean(ycord))


for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if (h ==r):
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if (update_potentialnodes[h]>update_potentialnodes[r] and h>0):
                if ((check(update_potentialnodes[r], update_potentialnodes[h]))):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r], distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                #add_edge(update_potentialnodes[h], update_potentialnodes[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))



#distance_matrix = [row[1:] for row in distance_matrix[1:]]


# Build a graph from pairwise node distances.
def create_graph_from_distance_matrix(distance_matrix):
    num_nodes = len(distance_matrix)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for node in range(1,num_nodes):
        G.add_node(node)
    
    # Add edges based on the distance matrix
    for i in range(1, num_nodes):
        for j in range(i+1, num_nodes):
            distance = distance_matrix[i][j]
            G.add_edge(i, j, weight=distance)
    
    return G



# Create the graph
G = create_graph_from_distance_matrix(distance_matrix)

nx.draw_networkx(G, with_labels = True, node_size = 500)


#for u, v in G.edges:
#    print(f"Edge: {u} -- {v}, Weight: {G[u][v]['weight']}")

# Construct a minimum spanning tree candidate.
def find_mst(G1):
    # Compute Minimum Spanning Tree (MST) of the given weighted graph
    return nx.minimum_spanning_tree(G1)

# Check whether the network satisfies its degree constraints.
def is_degree_satisfied(G1, d):
    # Check if all nodes in the graph satisfy the degree bound (degree <= d)
    for node in G1.nodes:
        if G1.degree(node) > d:
            return False
    return True

# Find edges incident to the selected node.
def get_edges_adjacent_to_node(G1, node):
    # Get the edges adjacent to the specified node
    adjacent_edges = []
    for u, v, data in G1.edges(data=True):
        if u == node or v == node:
            adjacent_edges.append((u, v, data['weight']))
    return adjacent_edges

# Identify candidate edge weights relative to the current threshold.
def find_weights_greater_than_least(edges):
    # Find the weights of all the edges greater than the least weight
    sorted_edges = sorted(edges, key=lambda x: x[2])
    least_weight = sorted_edges[0][2]
    return [edge[2] for edge in sorted_edges if edge[2] > least_weight]


# Build a spanning-tree candidate subject to degree restrictions.
def degree_constrained_mst_heuristics(G, d):
    # Start with the original graph as the current graph
    current_graph = copy.deepcopy(G)
    mst = find_mst(current_graph)
    # Repeat the process until all nodes satisfy the degree bound
    while not is_degree_satisfied(mst, d):
        # Check for nodes violating the degree constraint
        violating_nodes = [node for node in mst.nodes if mst.degree(node) > d]
        w_min = min(mst[u][v]['weight'] for u, v in mst.edges)
        w_max = max(mst[u][v]['weight'] for u, v in mst.edges)
        # Increase edge costs for violating nodes
        for node in violating_nodes:
            adjacent_edges = get_edges_adjacent_to_node(mst, node)
            weights_greater_than_least = find_weights_greater_than_least(adjacent_edges)
            #num_edges_to_increase = mst.degree(node) - d
            for u, v, weight in adjacent_edges:
                if weight in weights_greater_than_least:
                    current_graph[u][v]['weight'] += (len(violating_nodes))*((current_graph[u][v]['weight'] - w_min)/(w_max -w_min))*w_max
            #increase_edge_costs(current_graph, node, adjacent_edges, weights_greater_than_least, increment=1000)  # Adjust increment as needed
        #current_graph = G.copy()
        mst = find_mst(current_graph)
        
    return mst


if __name__ == "__main__":
    d = degree_node  # Replace 'd' with your desired degree bound

    # Apply degree-constrained MST heuristics
    result_mst = degree_constrained_mst_heuristics(G, d)

    
    # Print the result MST

    print("Result MST edges:")
    for u, v in result_mst.edges:
        print(f"Edge: {u} -- {v}, Weight: {G[u][v]['weight']}")
        
    edge1=[]
    for u, v in result_mst.edges:
        edge1.append((u,v))
            
# Initialize an empty dictionary to store the adjacency list
adjacency_list = {}

# Function to add edges to the adjacency list
def add_edge(node1, node2):
    if node1 not in adjacency_list:
        adjacency_list[node1] = []
    if node2 not in adjacency_list:
        adjacency_list[node2] = []
    adjacency_list[node1].append(node2)
    adjacency_list[node2].append(node1)  

for edge in edge1:
    node1, node2 = edge
    add_edge(node1, node2)

# Implement the update adjacency list routine.
def update_adjacency_list(adj_list, old_name, new_name):
    # Update the adjacency list to replace occurrences of the old node name with the new node name in the neighbors' lists
    for node, neighbors in adj_list.items():
        adj_list[node] = [new_name if n == old_name else n for n in neighbors]

degrees = {}

# Iterate through each node in the adjacency list
for node, neighbors in adjacency_list.items():
    degree = len(neighbors)
    degrees[node] = degree

#highest_degree_node = max(degrees, key=degrees.get)

highest_degree_node = rn.randint(1,numofnodes)


nodes_modified_track=[highest_degree_node]
adjacency_list[new_name] = adjacency_list.pop(highest_degree_node)
update_adjacency_list(adjacency_list, highest_degree_node, new_name)


connected_nodes = adjacency_list[new_name]
nodes_renamed=[]

# Implement the rename routine.
def rename(connected_nodes, new_name):
    
    if (len(connected_nodes) !=0):
        for node in connected_nodes:
            high_degree = node
            nodes_modified_track.append(high_degree)
            name = str(int(new_name) + 1)
            adjacency_list[name] = adjacency_list.pop(high_degree)
            #degrees[name] = degrees.pop(high_degree)
            update_adjacency_list(adjacency_list, high_degree, name)
            new_name = name
            nodes_renamed.append(name)
        connected = adjacency_list[nodes_renamed[0]]
        connected_nodes = [item for item in connected if isinstance(item, int)]
        new_name = str(int(nodes_renamed[len(nodes_renamed) -1]))
        nodes_renamed.remove(nodes_renamed[0])
        if (int(new_name) == numofnodes):
            return adjacency_list
        rename(connected_nodes, new_name)
    else:
        connected = adjacency_list[nodes_renamed[0]]
        connected_nodes = [item for item in connected if isinstance(item, int)]
        nodes_renamed.remove(nodes_renamed[0])
        if (int(new_name) == numofnodes):
            return adjacency_list
        rename(connected_nodes, new_name)
    return adjacency_list
            
s=rename(connected_nodes, new_name)

renamed_adjacency_list = {int(key): [int(val) for val in value] for key, value in s.items()}

minimum_spanning = nx.Graph(renamed_adjacency_list)

minimum_spanning_edge = list(minimum_spanning.edges())

minimum_spanning_tree = nx.DiGraph(minimum_spanning_edge)


# Now you have a NetworkX DiGraph representing the directed graph




# Implement the find graph depth dfs routine.
def find_graph_depth_dfs(graph, start_node):
    depth = 0
    visited = {node: False for node in graph.nodes}

    # Implement the dfs routine.
    def dfs(node, current_depth):
        nonlocal depth
        visited[node] = True
        depth = max(depth, current_depth)

        for neighbor in graph.neighbors(node):
            if not visited[neighbor]:
                dfs(neighbor, current_depth + 1)

    dfs(start_node, 1)
    return depth



# Implement the average outgoing degree directed routine.
def average_outgoing_degree_directed(graph):
    total_outgoing_degree = sum(graph.out_degree(node) for node in graph.nodes())
    num_nodes=[]
    for node in graph.nodes():
        if(graph.out_degree(node)!=0):
            num_nodes.append(node)

    if num_nodes == 0:
        return 0

    return total_outgoing_degree / len(num_nodes)


# Implement the min max outgoing degree routine.
def min_max_outgoing_degree(graph):
    out_degrees = [graph.out_degree(node) for node in graph.nodes()]
    
    if not out_degrees:
        return 0, 0  # If the graph has no nodes, both min and max outgoing degrees are 0
    
    min_out_degree = min(out_degrees)
    max_out_degree = max(out_degrees)
    
    return min_out_degree, max_out_degree

# Using DFS to find the depth of the graph
depth_dfs = find_graph_depth_dfs(minimum_spanning_tree, start_node)
print("Graph depth:", depth_dfs)
minimum_spanning_tree.out_degree(15)

avg_outgoing_degree_directed = average_outgoing_degree_directed(minimum_spanning_tree)
print("Average Outgoing Degree:", avg_outgoing_degree_directed)



min_out_degree, max_out_degree = min_max_outgoing_degree(minimum_spanning_tree)
print("Minimum Outgoing Degree:", min_out_degree)
print("Maximum Outgoing Degree:", max_out_degree)

network_name.append("Network_1")
depth_tree.append(depth_dfs)
avg_outgoing_degree.append(avg_outgoing_degree_directed)
min_outgoing_degree.append(min_out_degree)
max_outgoing_degree.append(max_out_degree)

predecessor =[]
line_node=[]
for u, v in minimum_spanning_tree.edges:
    print(f"Edge: {u} -- {v}, Weight: {G[u][v]['weight']}")
    predecessor.append(u)
    line_node.append(v)

   
predecessor.insert(0, 0)
line_node.insert(0, 1)    
    
#####Sorting of the coordiantes based on the renamed graph######

# Create a list of pairs (A, B) from the original arrays using zip
pairs = list(zip(line_network, sorted_X))

# Sort the pairs based on the new order of A
sorted_pairs_X = sorted(pairs, key=lambda x: nodes_modified_track.index(x[0]))

# Separate the sorted pairs back into separate A and B arrays
nodes_modified_track, new_sorted_X = zip(*sorted_pairs_X)

pairs = list(zip(line_network, sorted_Y))

# Sort the pairs based on the new order of A
sorted_pairs_Y = sorted(pairs, key=lambda x: nodes_modified_track.index(x[0]))

# Separate the sorted pairs back into separate A and B arrays
nodes_modified_track, new_sorted_Y = zip(*sorted_pairs_Y)



df_new = pd.DataFrame({'x':line_node, 'y':predecessor, 'xcord':new_sorted_X, 'ycord':new_sorted_Y, 'weight':weight})


with open(f"output_{numofnodes}.txt", 'w') as f:
    for index, row in df_new.iterrows():
        # Convert the first two columns to integers
        integer_values = '\t'.join(str(int(value)) for value in row[:2])
        # Convert the next two columns to floats
        float_values = '\t'.join(f'{value:.3f}' for value in row[2:4])
        # Convert the last column to an integer
        last_integer_value = int(row[-1])
        # Combine all values and write to the file
        f.write(f"{integer_values}\t{float_values}\t{last_integer_value}\n")
        
sns.scatterplot(x=new_sorted_X, y=new_sorted_Y, s=200, c='g')
for i in range(len(line_node)):
    plt.annotate(line_node[i], (new_sorted_X[i], new_sorted_Y[i]+ 1))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title(f"Location of {numofnodes} nodes")
plt.grid(True)
plt.xlim((0, rectangle_width + 5))
plt.ylim((0, rectangle_height + 5))
plt.xlabel("X-distance", fontsize=15)
plt.ylabel("Y-distance", fontsize=15)
plt.savefig(f"Location_plot_{numofnodes}.png")

plt.show()




f = open(f"output_{numofnodes}.txt",'r')

# Open the file in read mode ('r')
with f as file:
    # Read the lines of the file
    lines = file.readlines()


def add_vertex(v):
    global distance_matrix
    global vertices_no
    global vertices
    if v in vertices:
        print("Vertex ", v, " already exists")
    else:
        vertices_no = vertices_no + 1
    vertices.append(v)
    if vertices_no > 1:
        for vertex in distance_matrix:
            vertex.append(0)
    temp = []
    for i in range(vertices_no):
        temp.append(0)
    distance_matrix.append(temp)

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
    global distance_matrix
    global vertices_no
    global vertices
    # Check if vertex v1 is a valid vertex
    if v1 not in vertices:
        print("Vertex ", v1, " does not exist.")
    # Check if vertex v1 is a valid vertex
    elif v2 not in vertices:
        print("Vertex ", v2, " does not exist.")

    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        distance_matrix[index1][index2] = e


# Print the graph
def print_graph():
  global distance_matrix
  global vertices_no
  for i in range(vertices_no):
    for j in range(vertices_no):
      if distance_matrix[i][j] != 0:
        print(vertices[i], " -> ", vertices[j], \
        " edge weight: ", distance_matrix[i][j])


def check(i,j):
    if (distance_matrix[i][j]!=0):
        return True
    else:
        return False

vertices = []
# stores the number of vertices in the graph
vertices_no = 0
distance_matrix = []
# Add vertices to the graph

G_child = nx.DiGraph()

nodes = []
prede = []
xcord = []
ycord = []
weight =[]



for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    prede.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

    
edges = list(zip(prede, nodes[0:] + [None]))
del edges[0]
loca = list(zip(xcord, ycord[0:] + [None]))
node_positions = {key: value for key, value in zip(nodes, loca)}

G_child.add_edges_from(edges)
#fig, ax = plt.subplots()
#nx.draw_networkx(G_child, pos=node_positions, with_labels=True, node_size=200, node_color='skyblue', font_size=10, ax=ax)
# Draw edges between nodes

# Show the plot
fig, ax = plt.subplots(figsize=(16, 12))

edges_list = list(G_child.edges())
#plt.figure(figsize=(12, 8)) 

nx.draw_networkx_nodes(G_child, node_positions, cmap=plt.get_cmap('jet'), node_size=150, node_color='white', ax=ax)
nx.draw_networkx_labels(G_child, node_positions, font_size=22)
nx.draw_networkx_edges(G_child, node_positions, edgelist = edges_list, arrows=True)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=26)

#ax.tick_params(axis='both', which='major', labelsize=24) 

plt.xlabel("X-axis (Width of the region)", fontsize=26)  # Increase font size for X-axis label
plt.ylabel("Y-axis (height of the region)", fontsize=26)  # Increase font size for Y-axis label
plt.title("Power network", fontsize=26)  # Increase font size for the title

plt.savefig(f"Output_plot_{numofnodes}_{no_of_nodes_to_select}.png", dpi=600)

plt.show()
    
# Implement the orientation routine.
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or counterclockwise

# Implement the do segments cross routine.
def do_segments_cross(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    
    return False

   
    
# Implement the generate random not in array routine.
def generate_random_not_in_array(samples, lower_limit, upper_limit):
    while True:
        new_random_number = rn.randint(lower_limit, upper_limit)
        if new_random_number not in samples:
            return new_random_number

# Implement the cross check routine.
def cross_check(G1_child, G1_child_copy, x, y, s, g, n1):
    for m,n in G1_child.edges:        
        #print(m,n)
        if (m!=x and n!=y):
            p1=node_positions[y]
            q1=node_positions[x]
            p2=node_positions[m]
            q2=node_positions[n]
            if (do_segments_cross(p1, q1, p2, q2)):
                if (m==x or n==y or n==x or m==y):
                    pass
                    #G1_child_copy=copy.deepcopy(G1_child)
                else:
                    G1_child=copy.deepcopy(G1_child_copy)
                    unique_random = generate_random_not_in_array(s, n1[-1] + 1, numofnodes)
                    y=unique_random
                    #select_nodes.append(unique_random)
                    grand_parent_random = list(G_child.predecessors(list(G1_child.predecessors(unique_random))[0]))
                    x=grand_parent_random[0]
                    #grand_parent_node.append(grand_parent[0])
                    incoming_edges = list(G1_child.in_edges(unique_random))
                    if incoming_edges:
                        source_node, _ = incoming_edges[0]
                        G1_child.remove_edge(source_node, unique_random)
                    G1_child.add_edge(grand_parent_random[0], unique_random)
                    cross_check(G1_child, G1_child_copy, grand_parent_random[0], unique_random, s, g, n1)
            else:
                pass
                #G1_child_copy=copy.deepcopy(G1_child)
        else:
            pass
            #G1_child_copy=copy.deepcopy(G1_child)
    return G1_child, y, x
        
    # Get the descendants of the node

##################################################################################

for i in range(1,count_modify):
    f = open(f"output_{numofnodes}.txt",'r')
    
    # Open the file in read mode ('r')
    with f as file:
        # Read the lines of the file
        lines = file.readlines()
    
    
    def add_vertex(v):
        global distance_matrix
        global vertices_no
        global vertices
        if v in vertices:
            print("Vertex ", v, " already exists")
        else:
            vertices_no = vertices_no + 1
        vertices.append(v)
        if vertices_no > 1:
            for vertex in distance_matrix:
                vertex.append(0)
        temp = []
        for i in range(vertices_no):
            temp.append(0)
        distance_matrix.append(temp)
    
    # Add an edge between vertex v1 and v2 with edge weight e
    def add_edge(v1, v2, e):
        global distance_matrix
        global vertices_no
        global vertices
        # Check if vertex v1 is a valid vertex
        if v1 not in vertices:
            print("Vertex ", v1, " does not exist.")
        # Check if vertex v1 is a valid vertex
        elif v2 not in vertices:
            print("Vertex ", v2, " does not exist.")
    
        else:
            index1 = vertices.index(v1)
            index2 = vertices.index(v2)
            distance_matrix[index1][index2] = e
    
    
    # Print the graph
    def print_graph():
      global distance_matrix
      global vertices_no
      for i in range(vertices_no):
        for j in range(vertices_no):
          if distance_matrix[i][j] != 0:
            print(vertices[i], " -> ", vertices[j], \
            " edge weight: ", distance_matrix[i][j])
    
    
    def check(i,j):
        if (distance_matrix[i][j]!=0):
            return True
        else:
            return False
    
    vertices = []
    # stores the number of vertices in the graph
    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    G_child = nx.DiGraph()
    
    nodes = []
    prede = []
    xcord = []
    ycord = []
    weight =[]
    
    
    
    for line in lines:
        values = line.strip().split()
        nodes.append(int(values[0]))
        prede.append(int(values[1]))
        xcord.append(float(values[2]))
        ycord.append(float(values[3]))
        weight.append(int(values[4]))
    
        
    edges = list(zip(prede, nodes[0:] + [None]))
    del edges[0]
    loca = list(zip(xcord, ycord[0:] + [None]))
    node_positions = {key: value for key, value in zip(nodes, loca)}
    
    
    
    
    G_child.add_edges_from(edges)
    nodes_to_1 = list(G_child.successors(1))
    select_nodes=list(rn.sample(range(nodes_to_1[-1] + 1, numofnodes+1),no_of_nodes_to_select))

    select_nodes.sort(reverse=True)
    print(select_nodes)
    grand_parent_node=[]
    #snode_deleted=[]
    for j in select_nodes:
        grand_parent = list(G_child.predecessors(list(G_child.predecessors(j))[0]))
        grand_parent_node.append(grand_parent[0])
        #if (len(grand_parent)==0):
        #    print("Doesn't have a grandparent, select a different node")
        #    while True:
        #        new_select_node = rn.randint(3, numofnodes)
        #        if new_select_node not in select_nodes:
        #            break
        #    select_nodes.append(new_select_node)
        #    snode_deleted.append(j)
        #    grand_parent = list(G_child.predecessors(list(G_child.predecessors(new_select_node))[0]))
        #    grand_parent_node.append(grand_parent[0])
        #else:
        #    grand_parent_node.append(grand_parent[0])
    #if (len(snode_deleted)>0):
    #    select_nodes.remove(snode_deleted[0])
    #    grand_parent_node.pop(0)
    print(grand_parent_node)
    #select_nodes = [18, 15, 14, 7]
    #grand_parent_node = [13, 11, 11, 3]
    
    for k in range(len(select_nodes)):
        G_child_copy=copy.deepcopy(G_child)
        incoming_edges = list(G_child.in_edges(select_nodes[k]))
        if incoming_edges:
            source_node, _ = incoming_edges[0]
            G_child.remove_edge(source_node, select_nodes[k])
        G_child.add_edge(grand_parent_node[k], select_nodes[k])
        
        G_child, new_node, grandparent_new_node = cross_check(G_child, G_child_copy, grand_parent_node[k], select_nodes[k], select_nodes, grand_parent_node, nodes_to_1)
        print(new_node)
        print(grandparent_new_node)
   
    Nodes_original = []
    Nodes_predecessor = []
    
    # Iterate through the edges and store the source and target nodes in the arrays
    for source, target in G_child.edges:
        Nodes_predecessor.append(source)
        Nodes_original.append(target)
    
    # Sort the left and right nodes arrays together based on the right nodes
    Nodes_predecessor, Nodes_original = zip(*sorted(zip(Nodes_predecessor, Nodes_original), key=lambda x: x[1]))
    
        
    Nodes_predecessor = list(Nodes_predecessor)
    Nodes_original=list(Nodes_original)
    
    Nodes_predecessor.insert(0, 0)
    Nodes_original.insert(0, 1)

#fig, ax = plt.subplots()
#nx.draw_networkx(G_child, pos=node_positions, with_labels=True, node_size=150, node_color='skyblue', font_size=10, ax=ax)
# Draw edges between nodes

# Show the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    edges_list = list(G_child.edges())
    #plt.figure(figsize=(12, 8)) 
    
    nx.draw_networkx_nodes(G_child, node_positions, cmap=plt.get_cmap('jet'), node_size=150, node_color='white', ax=ax)
    nx.draw_networkx_labels(G_child, node_positions, font_size=22)
    nx.draw_networkx_edges(G_child, node_positions, edgelist = edges_list, arrows=True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=26)
    
    #ax.tick_params(axis='both', which='major', labelsize=24) 
    
    plt.xlabel("X-axis (Width of the region)", fontsize=26)  # Increase font size for X-axis label
    plt.ylabel("Y-axis (height of the region)", fontsize=26)  # Increase font size for Y-axis label
    plt.title("Power network", fontsize=26)  # Increase font size for the title
    
    plt.savefig(f"Output_plot_{numofnodes}_{no_of_nodes_to_select}.png", dpi=600)
    
    plt.show()    
    df_child = pd.DataFrame({'x':Nodes_original, 'y':Nodes_predecessor, 'xcord':new_sorted_X, 'ycord':new_sorted_Y, 'weight':weight})
    
    
    with open(f"output_{numofnodes}_{i}_{no_of_nodes_to_select}.txt", 'w') as f:
        for index, row in df_child.iterrows():
            # Convert the first two columns to integers
            integer_values = '\t'.join(str(int(value)) for value in row[:2])
            # Convert the next two columns to floats
            float_values = '\t'.join(f'{value:.3f}' for value in row[2:4])
            # Convert the last column to an integer
            last_integer_value = int(row[-1])
            # Combine all values and write to the file
            f.write(f"{integer_values}\t{float_values}\t{last_integer_value}\n")
    
    depth_dfs_new = find_graph_depth_dfs(G_child, start_node)
    print("Graph depth:", depth_dfs_new)
    
    
    avg_outgoing_degree_directed_new = average_outgoing_degree_directed(G_child)
    print("Average Outgoing Degree:", avg_outgoing_degree_directed_new)
    
    
    min_out_degree_new, max_out_degree_new = min_max_outgoing_degree(G_child)
    print("Minimum Outgoing Degree:", min_out_degree_new)
    print("Maximum Outgoing Degree:", max_out_degree_new)
    
    network_name.append(f"Network_{i}")
    depth_tree.append(depth_dfs_new)
    avg_outgoing_degree.append(avg_outgoing_degree_directed_new)
    min_outgoing_degree.append(min_out_degree_new)
    max_outgoing_degree.append(max_out_degree_new)
    
depth_degree = pd.DataFrame({'Network Name':network_name, 'Depth':depth_tree, 'Average Outgoing Degree':avg_outgoing_degree, 'Min Outgoing Degree':min_outgoing_degree, 'Max Outgoing Degree':max_outgoing_degree})
print(depth_degree)
depth_degree.to_csv(f"Table_{numofnodes}_{no_of_nodes_to_select}.csv")




