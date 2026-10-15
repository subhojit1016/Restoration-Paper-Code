# ==========================================================================
# K_Step NN.py
# Purpose: Evaluate k-step nearest-neighbor restoration heuristics across experimental settings.
# Publication note: Original executable statements, names, and parameters are unchanged.
# Inputs and outputs: Check the file paths and experiment settings below before running.
# ==========================================================================

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

# Register a vertex in the network representation.
def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

# Add an edge to the network representation.
def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

# Check a condition used by the algorithm.
def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
# Simulate candidate restoration actions over a k-step lookahead.
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

# Choose the next action using the k-step evaluation.
def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.90_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.90_0.0_19June2026.csv", index=False)   # simple CSV




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 0.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.25_0.0_19June2026.csv", index=False)   # simple CSV




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 0.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.50_0.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 0.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.75_0.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 3.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.90_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.90_3.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 3.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.25_3.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 3.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.50_3.0_19June2026.csv", index=False)   # simple CSV



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 3.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.75_3.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 0.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.90_0.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 0.0
k_steps = 25 # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.25_0.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 0.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.50_0.0_19June2026.csv", index=False)   # simple CSV



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 0.0
k_steps = 25  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network50_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network50.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network50_Cycle_0.75_0.0_19June2026.csv", index=False)   # simple CSV



##############################################################

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.90_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.90_0.0_19June2026.csv", index=False)   # simple CSV




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.25_0.0_19June2026.csv", index=False)   # simple CSV




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.50_0.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.75_0.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 3.0
k_steps = 35  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.90_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.90_3.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 3.0
k_steps = 35  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.25_3.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 3.0
k_steps = 35  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.50_3.0_19June2026.csv", index=False)   # simple CSV



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 3.0
k_steps = 35  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.75_3.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 1.5
k_steps = 35  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.90_1.5_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 1.5
k_steps = 35 # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.25_1.5_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 1.5
k_steps = 35 # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.50_1.5_19June2026.csv", index=False)   # simple CSV



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 1.5
k_steps = 35 # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network70_Cycle_0.75_1.5_19June2026.csv", index=False)   # simple CSV

##################################################

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.90_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.90_0.0_19June2026.csv", index=False)   # simple CSV




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.25_0.0_19June2026.csv", index=False)   # simple CSV




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.50_0.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.75_0.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.90_3.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 0.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.25_3.0_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 3.0
k_steps = 35  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network70.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network70.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.50_3.0_19June2026.csv", index=False)   # simple CSV



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 3.0
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.75_3.0_19June2026.csv", index=False)   # simple CSV





import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.90
service = 1.5
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.90_1.5_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.25
service = 1.5
k_steps = 50 # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.25_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.25_1.5_19June2026.csv", index=False)   # simple CSV


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.50
service = 1.5
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.50_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.50_1.5_19June2026.csv", index=False)   # simple CSV



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial import distance
import pandas as pd
import ast

# -------------------
# Parameters
# -------------------
prob_value = 0.75
service = 1.5
k_steps = 50  # Change this to 1 for standard NN

# -------------------
# Graph Setup
# -------------------
vertices = []
vertices_no = 0
distance_matrix = []

def add_vertex(v):
    global distance_matrix, vertices_no, vertices
    if v not in vertices:
        vertices_no += 1
        vertices.append(v)
        for vertex in distance_matrix:
            vertex.append(0)
        temp = [0] * vertices_no
        distance_matrix.append(temp)

def add_edge(v1, v2, e):
    global distance_matrix, vertices
    if v1 not in vertices or v2 not in vertices:
        print("Invalid vertex:", v1, v2)
        return
    index1 = vertices.index(v1)
    index2 = vertices.index(v2)
    distance_matrix[index1][index2] = e

def check(i, j):
    return distance_matrix[i][j] != 0

# Load network
G = nx.DiGraph()
nodes, predecessor, xcord, ycord, weight = [], [], [], [], []
filename = "Network100_Cycle.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    nodes.append(int(values[0]))
    predecessor.append(int(values[1]))
    xcord.append(float(values[2]))
    ycord.append(float(values[3]))
    weight.append(int(values[4]))

nodes_updated = copy.deepcopy(nodes)
predecessor_update = copy.deepcopy(predecessor)
nodes_updated.remove(1)
predecessor_update.remove(0)

for i, j in zip(nodes_updated, predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes = list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
update_potentialnodes.insert(0, 0)

for h in range(len(update_potentialnodes)):
    add_vertex(update_potentialnodes[h])

xcord.insert(0, np.mean(xcord))
ycord.insert(0, np.mean(ycord))

for h in range(len(update_potentialnodes)):
    for r in range(len(update_potentialnodes)):
        if h == r:
            add_edge(update_potentialnodes[h], update_potentialnodes[r], 0)
        else:
            if update_potentialnodes[h] > update_potentialnodes[r] and h > 0:
                if check(update_potentialnodes[r], update_potentialnodes[h]):
                    add_edge(update_potentialnodes[h], update_potentialnodes[r],
                             distance_matrix[update_potentialnodes[r]][update_potentialnodes[h]])
            else:
                add_edge(update_potentialnodes[h], update_potentialnodes[r],
                         distance.euclidean((xcord[h], ycord[h]), (xcord[r], ycord[r])))

# -------------------
# K-step NN Functions
# -------------------
def simulate_k_steps(start_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Simulate up to k NN steps from the current state with precedence updates."""
    curr_node = start_node
    sim_wp = copy.deepcopy(wp)
    sim_wop = copy.deepcopy(wop_visited)
    sim_notfaulty = copy.deepcopy(notfaulty)
    visited_nodes = sim_wp + sim_wop
    total_cost = 0

    for step in range(k):
        remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
        if not remaining_nodes:
            break

        nearest_node = min(remaining_nodes, key=lambda x: dist_matrix[curr_node][x])

        if nearest_node in sim_notfaulty:
            step_cost = dist_matrix[curr_node][nearest_node] * (len(all_nodes) - len(sim_wp))
        else:
            step_cost = (dist_matrix[curr_node][nearest_node] + service) * (len(all_nodes) - len(sim_wp))
        total_cost += step_cost

        all_pre_n = list(nx.ancestors(G, nearest_node))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(nearest_node)
        else:
            sim_wop.append(nearest_node)

        curr_node = nearest_node
        visited_nodes = sim_wp + sim_wop

        to_delete = []
        for cand in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand not in sim_wp:
                    sim_wp.append(cand)
                if cand in sim_wop:
                    sim_wop.remove(cand)
                to_delete.append(cand)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

    return total_cost

def choose_best_next_k_step(current_node, wp, wop_visited, notfaulty, all_nodes, dist_matrix, k, G):
    """Choose the best next node using k-step NN lookahead."""
    visited_nodes = wp + wop_visited
    remaining_nodes = [n for n in all_nodes if n not in visited_nodes]
    if not remaining_nodes:
        return None

    if k == 1:
        return min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])

    best_node, best_cost = None, float('inf')
    for cand in remaining_nodes:
        sim_wp = copy.deepcopy(wp)
        sim_wop = copy.deepcopy(wop_visited)
        sim_notfaulty = copy.deepcopy(notfaulty)

        if cand in sim_notfaulty:
            cost_first = dist_matrix[current_node][cand] * (len(all_nodes) - len(sim_wp))
        else:
            cost_first = (dist_matrix[current_node][cand] + service) * (len(all_nodes) - len(sim_wp))

        all_pre_n = list(nx.ancestors(G, cand))
        if all(item in sim_wp for item in all_pre_n):
            sim_wp.append(cand)
        else:
            sim_wop.append(cand)

        to_delete = []
        for cand_nf in sim_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand_nf))
            if all(item in sim_wp for item in all_pre_n_check):
                if cand_nf not in sim_wp:
                    sim_wp.append(cand_nf)
                if cand_nf in sim_wop:
                    sim_wop.remove(cand_nf)
                to_delete.append(cand_nf)
        sim_notfaulty = [x for x in sim_notfaulty if x not in to_delete]

        cost_future = simulate_k_steps(cand, sim_wp, sim_wop, sim_notfaulty, all_nodes, dist_matrix, k-1, G)
        total_cost = cost_first + cost_future

        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand

    return best_node

# -------------------
# Run for each realization
# -------------------

data_result = []
list_realization = pd.read_csv('Realization_0.75_Network100.csv')
realization = list(list_realization['realization'])

for r1 in range(len(realization)):
    faulty_nodes = ast.literal_eval(realization[r1])
    start_node = copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in faulty_nodes]
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)

    State = [[], [1], [], start_node]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n = []
    action_near = []

    for _ in range(len(potential_nodes)):
        next_node = choose_best_next_k_step(action_n, nodes_wp_n, nodes_wop_visited_n,
                                            N_nodes_notfaulty, potential_nodes,
                                            distance_matrix, k_steps, G)
        if next_node is None:
            break

        if next_node in nodes_notfaulty:
            cost_near = distance_matrix[action_n][next_node] * (len(potential_nodes) - len(nodes_wp_n))
        else:
            cost_near = (distance_matrix[action_n][next_node] + service) * (len(potential_nodes) - len(nodes_wp_n))

        cost_n.append(cost_near)
        action_near.append(next_node)

        all_pre_n = list(nx.ancestors(G, next_node))
        if all(item in nodes_wp_n for item in all_pre_n):
            nodes_wp_n.append(next_node)
        else:
            nodes_wop_visited_n.append(next_node)

        action_n = next_node

        to_delete = []
        for cand in N_nodes_notfaulty:
            all_pre_n_check = list(nx.ancestors(G, cand))
            if all(item in nodes_wp_n for item in all_pre_n_check):
                if cand not in nodes_wp_n:
                    nodes_wp_n.append(cand)
                if cand in nodes_wop_visited_n:
                    nodes_wop_visited_n.remove(cand)
                elif cand in nodes_wop_n:
                    nodes_wop_n.remove(cand)
                to_delete.append(cand)
        N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in to_delete]

    print(f"Realization {r1+1}:")
    print("  Policy:", action_near)
    print("  Costs:", cost_n)
    print("  Total cost:", sum(cost_n))
    # If you ALSO compute a priority policy elsewhere, include these:
# action_priority = ...   # list of nodes (priority sequence)
# cost_p = [...]          # list of step costs for priority
# pol = ...               # if you use this in your code
# cost = ...              # if you use this in your code

# Build one row for this realization
    row = [
        ast.literal_eval(realization[r1]),   # faulty_nodes
        prob_value,                          # prob value you set up top
        action_near,                         # NN / k-step policy
        float(sum(cost_n))                  # NN total cost
    ]
    data_result.append(row)

df = pd.DataFrame(data_result)
df.to_csv("Network100_Cycle_0.75_1.5_19June2026.csv", index=False)   # simple CSV



