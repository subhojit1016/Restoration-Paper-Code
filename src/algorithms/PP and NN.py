# ==========================================================================
# PP and NN.py
# Purpose: Evaluate priority-based and nearest-neighbor restoration heuristics.
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
import ast


prob_value = 0.25
service = 0.0




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

G = nx.DiGraph()

nodes = []
predecessor = []
xcord = []
ycord = []
weight =[]
  
filename = "Network20.txt"
f = open(filename,'r')

with f as file:
    # Read the lines of the file
    lines = file.readlines()

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

for i,j in zip (nodes_updated,predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes=list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
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


nx.draw_networkx(G, with_labels = True, node_size = 100)




rows, cols = (1, 8)
data_result = [[0 for i in range(cols)] for j in range(rows)]


list_realization = pd.read_csv('Realization_0.25_Network20.csv')
realization = list(list_realization['realization'])       
    
for r1 in range(len(realization)):    
    print("Realization " + str(ast.literal_eval(realization[r1])))
    rows, cols = (1, 8)
    dynamic = [[0 for k in range(cols)] for j in range(rows)]
    cost = []
    start_node= copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in ast.literal_eval(realization[r1])]
    #nodes_notfaulty = [ele for ele in start_node if ele not in K]
    copy_nodes_notfaulty=copy.deepcopy(nodes_notfaulty)
    pol = 0
    cost = 0
            
        
    #print("The optimal policy is " + str(pol))
    #p#rint("The cost without is " + str(cost))
    
    #print("The cost is " + str(sum(cost)))
    
    ##############################################################################
    
    # Evaluate a nearest-node selection rule.
    def nearest2(vec,k1,distance_matrix,G2, vec1):
        distance1=[]
        for j2 in range(len(vec)):   
            modified_distance = (distance_matrix[k1][vec[j2]])            
            distance1.append(modified_distance)
        #print(distance1)
        if (len(distance1)==0):
            res=[]
            temp = 0
        else:
            temp = min(distance1)
            #res = [i for i, j in enumerate(distance1) if j == temp]
            res = distance1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
    
        if (len(vec)!=0):
            nearest2(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    initial_u2 = potential_nodes
    initial_u1 = copy.deepcopy(initial_u2)
    route=[]
    #for i in range(len(potential_nodes)):
    #    if i != s2:
    #        initial_u1.append(i)
    

    initial_nodes = copy.deepcopy(initial_u1)
    
    initial_route = nearest2(initial_u1, 0, distance_matrix, G,route)
    
    

        
    start_node_n = copy.deepcopy(start_node)
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State = [[],[1],[],start_node_n]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n=[]
    action_near=[]
    nodes_visited = nodes_wp_n + nodes_wop_visited_n
    #remaining_nodes=copy.deepcopy(realization)
    for l in range(len(initial_route)):
        GG=[]
        if (initial_route[l] in nodes_visited):
            print("Go to next node")
        else:
            
            if (initial_route[l] in nodes_notfaulty):
                cost_near = (distance_matrix[action_n][initial_route[l]])*(len(initial_nodes) - len(nodes_wp_n))
            else:
                cost_near = (distance_matrix[action_n][initial_route[l]] + service)*(len(initial_nodes) - len(nodes_wp_n))
            all_pre_n = list(nx.ancestors(G, initial_route[l])) 
            if (all(item in nodes_wp_n for item in all_pre_n)):
                nodes_wp_n.append(initial_route[l])
            else:
                nodes_wop_visited_n.append(initial_route[l])
            
            action_n = initial_route[l]
            cost_n.append(cost_near)
            action_near.append(action_n)
            nodes_tobe_deleted_n =[]
            for k in range(len(N_nodes_notfaulty)):
                all_pre_n_check = list(nx.ancestors(G, N_nodes_notfaulty[k]))
                if (all(item in nodes_wp_n for item in all_pre_n_check)):
                    if (N_nodes_notfaulty[k] in nodes_wp_n):
                        nodes_wp_n = nodes_wp_n
                    else:
                        nodes_wp_n.append(N_nodes_notfaulty[k])
                    if (N_nodes_notfaulty[k] in nodes_wop_visited_n):
                        nodes_wop_visited_n.remove(N_nodes_notfaulty[k])
                    elif (N_nodes_notfaulty[k] in nodes_wop_n):
                        nodes_wop_n.remove(N_nodes_notfaulty[k])
                    nodes_tobe_deleted_n.append(N_nodes_notfaulty[k])
            N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in nodes_tobe_deleted_n]
        #    print(N_nodes_notfaulty, nodes_wop_visited_n,nodes_wp_n)
            #extra_nodes = succ_prec(action_n, nodes_wop_visited_n,nodes_wp_n,N_nodes_notfaulty, GG)
        #    print(extra_nodes)
            nodes_visited = nodes_wp_n + nodes_wop_visited_n
            remaining_nodes =[ele for ele in initial_route if ele not in nodes_visited]
        #    print(remaining_nodes)

    
    
    print("The optimal policy for nearest neighbour is " + str(action_near))
    print("The cost for each action is " + str(cost_n))
    
    print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    ##############################################################################
    
      
  
    # Compute a distance-related priority measure.
    def priority_dist(vec,k1,distance_matrix,G2, vec1):
        priority1=[]
        for j2 in range(len(vec)):   
            modified = (distance_matrix[k1][vec[j2]])            
            priority1.append(modified)
        #print(priority1)
        if (len(priority1)==0):
            res=[]
            temp = 0
        else:
            temp = max(priority1)
            res = priority1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
        if (len(vec)!=0):
            priority_dist(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    
    # Compute the priority used to rank candidate actions.
    def prior(G2, pri):
        prio=[]
        for i in range(len(pri)):
            dec = list(nx.descendants(G2, i))
            prio.append(len(dec))
        #print(prio)
        updated_priority=prio*pri
        return updated_priority   
    
    vertices = []
    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    
    G2 = nx.DiGraph()
    
    nodes = []
    predecessor = []
    xcord = []
    ycord = []
    weight =[]
      
    f = open(filename,'r')

    with f as file:
        # Read the lines of the file
        lines = file.readlines()
    
    for line in lines:
        values = line.strip().split()
        nodes.append(int(values[0]))
        predecessor.append(int(values[1]))
        xcord.append(float(values[2]))
        ycord.append(float(values[3]))
        weight.append(int(values[4]))
        
    for i,j in zip (nodes,predecessor):
        G2.add_edges_from([(j, i)])
    
    nx.draw_networkx(G2, with_labels = True, node_size = 500)
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    xcord.insert(0,np.mean(xcord))
    ycord.insert(0,np.mean(ycord))

    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], distance_matrix[potential_nodes_p[r]][potential_nodes_p[h]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    #add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    
    
    
    distance_matrix_3=distance_matrix
   
    vertices = []

    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    
    priority=[1 for element in range(len(potential_nodes)+1)]
    updtd_prior = np.true_divide(priority, sum(priority))
    updated_priority_val = prior(G2, updtd_prior)
    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                    #add_edge(potential_nodes[h], potential_nodes[r], random.randint(20, 100))
    
    
    
    
    distance_matrix1=distance_matrix
    distance_matrix=distance_matrix_3
    
    
    initial_u1 = copy.deepcopy(potential_nodes)
    route=[]
    
    initial_nodes = copy.deepcopy(potential_nodes)
    initial_route_p = priority_dist(initial_u1, 0, distance_matrix1, G2,route)
            
    start_node_p = copy.deepcopy(start_node)
    p_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State_p = [[],[1],[],start_node_p]
    action_p = 0
    nodes_wp_p = []
    nodes_wop_visited_p = []
    nodes_wop_p = State_p[1] + State_p[3]
    cost_p=[]
    action_priority=[]
    nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
    for l in range(len(initial_route_p)):
        GG_p=[]
        if (initial_route_p[l] in nodes_visited_p):
            print("Priority Go to next node")
        else:
            if (initial_route_p[l] in nodes_notfaulty):
                
                cost_priority = (distance_matrix[action_p][initial_route_p[l]])*(len(initial_nodes) - len(nodes_wp_p))
            else:
                cost_priority = (distance_matrix[action_p][initial_route_p[l]] + service)*(len(initial_nodes) - len(nodes_wp_p))
            all_pre_p = list(nx.ancestors(G, initial_route_p[l]))
            if (all(item in nodes_wp_p for item in all_pre_p)):
                nodes_wp_p.append(initial_route_p[l])
            else:
                nodes_wop_visited_p.append(initial_route_p[l])
            
            action_p = initial_route_p[l]
            cost_p.append(cost_priority)
            action_priority.append(action_p)
            nodes_tobe_deleted_p =[]
            for k in range(len(p_nodes_notfaulty)):
                all_pre_p_check = list(nx.ancestors(G, p_nodes_notfaulty[k]))
                if (all(item in nodes_wp_p for item in all_pre_p_check)):
                    if (p_nodes_notfaulty[k] in nodes_wp_p):
                        nodes_wp_p = nodes_wp_p
                    else:
                        nodes_wp_p.append(p_nodes_notfaulty[k])
                    if (p_nodes_notfaulty[k] in nodes_wop_visited_p):
                        nodes_wop_visited_p.remove(p_nodes_notfaulty[k])
                    elif (p_nodes_notfaulty[k] in nodes_wop_p):
                        nodes_wop_p.remove(p_nodes_notfaulty[k])
                    nodes_tobe_deleted_p.append(p_nodes_notfaulty[k])
            p_nodes_notfaulty = [ele for ele in p_nodes_notfaulty if ele not in nodes_tobe_deleted_p]

            
            #extra_nodes_p = succ_prec(action_p, nodes_wop_visited_p,nodes_wp_p, p_nodes_notfaulty, GG_p)
            nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
            print("with power", nodes_wp_p)
            remaining_nodes_p =[ele for ele in initial_route_p if ele not in nodes_visited_p] 
    
    
    
    #print("Realization " + str(ast.literal_eval(realization[r1])))
    nodesnotfaulty = [ele for ele in potential_nodes if ele not in ast.literal_eval(realization[r1])]
    probval = pow(prob_value, len(ast.literal_eval(realization[r1]))) * pow((1- prob_value), len(nodesnotfaulty))
    
    #print("The optimal policy for priority algorithm is " + str(action_priority))
    #print("The cost for each action is " + str(cost_p))
    
    #print("The cost for priority algorithm is " + str(sum(cost_p)))
    
    
    #print("The optimal policy for nearest neighbour is " + str(action_near))
    #print("The cost for each action is " + str(cost_n))
    
    #print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    #print("The optimal policy is " + str(pol))
    #print("The cost without is " + str(cost))
    
    
    #print("The cost is " + str(sum(cost)))
    dynamic[0][0] = ast.literal_eval(realization[r1])
    dynamic[0][1] = probval
    dynamic[0][2] = action_priority
    dynamic[0][3] = sum(cost_p)
    dynamic[0][4] = action_near
    dynamic[0][5] = sum(cost_n)
    dynamic[0][6] = pol
    dynamic[0][7] = cost
    
    data_result = np.array(data_result, dtype=object)
    dynamic = np.array(dynamic, dtype=object)
    data_result = np.vstack([data_result, dynamic])
    
dat = pd.DataFrame(data_result, index=None)
dat.to_csv("20_test_14June2026.csv", index=False, header=False)






prob_value = 0.50
service = 0.0




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

G = nx.DiGraph()

nodes = []
predecessor = []
xcord = []
ycord = []
weight =[]
  
filename = "Network100_Real.txt"
f = open(filename,'r')

with f as file:
    # Read the lines of the file
    lines = file.readlines()

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

for i,j in zip (nodes_updated,predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes=list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
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


nx.draw_networkx(G, with_labels = True, node_size = 100)




rows, cols = (1, 8)
data_result = [[0 for i in range(cols)] for j in range(rows)]


list_realization = pd.read_csv('Realization_0.50_Network100.csv')
realization = list(list_realization['realization'])       
    
for r1 in range(len(realization)):    
    print("Realization " + str(ast.literal_eval(realization[r1])))
    rows, cols = (1, 8)
    dynamic = [[0 for k in range(cols)] for j in range(rows)]
    cost = []
    start_node= copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in ast.literal_eval(realization[r1])]
    #nodes_notfaulty = [ele for ele in start_node if ele not in K]
    copy_nodes_notfaulty=copy.deepcopy(nodes_notfaulty)
    pol = 0
    cost = 0
            
        
    #print("The optimal policy is " + str(pol))
    #p#rint("The cost without is " + str(cost))
    
    #print("The cost is " + str(sum(cost)))
    
    ##############################################################################
    
    def nearest2(vec,k1,distance_matrix,G2, vec1):
        distance1=[]
        for j2 in range(len(vec)):   
            modified_distance = (distance_matrix[k1][vec[j2]])            
            distance1.append(modified_distance)
        #print(distance1)
        if (len(distance1)==0):
            res=[]
            temp = 0
        else:
            temp = min(distance1)
            #res = [i for i, j in enumerate(distance1) if j == temp]
            res = distance1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
    
        if (len(vec)!=0):
            nearest2(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    initial_u2 = potential_nodes
    initial_u1 = copy.deepcopy(initial_u2)
    route=[]
    #for i in range(len(potential_nodes)):
    #    if i != s2:
    #        initial_u1.append(i)
    

    initial_nodes = copy.deepcopy(initial_u1)
    
    initial_route = nearest2(initial_u1, 0, distance_matrix, G,route)
    
    

        
    start_node_n = copy.deepcopy(start_node)
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State = [[],[1],[],start_node_n]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n=[]
    action_near=[]
    nodes_visited = nodes_wp_n + nodes_wop_visited_n
    #remaining_nodes=copy.deepcopy(realization)
    for l in range(len(initial_route)):
        GG=[]
        if (initial_route[l] in nodes_visited):
            print("Go to next node")
        else:
            
            if (initial_route[l] in nodes_notfaulty):
                cost_near = (distance_matrix[action_n][initial_route[l]])*(len(initial_nodes) - len(nodes_wp_n))
            else:
                cost_near = (distance_matrix[action_n][initial_route[l]] + service)*(len(initial_nodes) - len(nodes_wp_n))
            all_pre_n = list(nx.ancestors(G, initial_route[l])) 
            if (all(item in nodes_wp_n for item in all_pre_n)):
                nodes_wp_n.append(initial_route[l])
            else:
                nodes_wop_visited_n.append(initial_route[l])
            
            action_n = initial_route[l]
            cost_n.append(cost_near)
            action_near.append(action_n)
            nodes_tobe_deleted_n =[]
            for k in range(len(N_nodes_notfaulty)):
                all_pre_n_check = list(nx.ancestors(G, N_nodes_notfaulty[k]))
                if (all(item in nodes_wp_n for item in all_pre_n_check)):
                    if (N_nodes_notfaulty[k] in nodes_wp_n):
                        nodes_wp_n = nodes_wp_n
                    else:
                        nodes_wp_n.append(N_nodes_notfaulty[k])
                    if (N_nodes_notfaulty[k] in nodes_wop_visited_n):
                        nodes_wop_visited_n.remove(N_nodes_notfaulty[k])
                    elif (N_nodes_notfaulty[k] in nodes_wop_n):
                        nodes_wop_n.remove(N_nodes_notfaulty[k])
                    nodes_tobe_deleted_n.append(N_nodes_notfaulty[k])
            N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in nodes_tobe_deleted_n]
        #    print(N_nodes_notfaulty, nodes_wop_visited_n,nodes_wp_n)
            #extra_nodes = succ_prec(action_n, nodes_wop_visited_n,nodes_wp_n,N_nodes_notfaulty, GG)
        #    print(extra_nodes)
            nodes_visited = nodes_wp_n + nodes_wop_visited_n
            remaining_nodes =[ele for ele in initial_route if ele not in nodes_visited]
        #    print(remaining_nodes)

    
    
    print("The optimal policy for nearest neighbour is " + str(action_near))
    print("The cost for each action is " + str(cost_n))
    
    print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    ##############################################################################
    
      
  
    def priority_dist(vec,k1,distance_matrix,G2, vec1):
        priority1=[]
        for j2 in range(len(vec)):   
            modified = (distance_matrix[k1][vec[j2]])            
            priority1.append(modified)
        #print(priority1)
        if (len(priority1)==0):
            res=[]
            temp = 0
        else:
            temp = max(priority1)
            res = priority1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
        if (len(vec)!=0):
            priority_dist(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    
    def prior(G2, pri):
        prio=[]
        for i in range(len(pri)):
            dec = list(nx.descendants(G2, i))
            prio.append(len(dec))
        #print(prio)
        updated_priority=prio*pri
        return updated_priority   
    
    vertices = []
    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    
    G2 = nx.DiGraph()
    
    nodes = []
    predecessor = []
    xcord = []
    ycord = []
    weight =[]
      
    f = open(filename,'r')

    with f as file:
        # Read the lines of the file
        lines = file.readlines()
    
    for line in lines:
        values = line.strip().split()
        nodes.append(int(values[0]))
        predecessor.append(int(values[1]))
        xcord.append(float(values[2]))
        ycord.append(float(values[3]))
        weight.append(int(values[4]))
        
    for i,j in zip (nodes,predecessor):
        G2.add_edges_from([(j, i)])
    
    nx.draw_networkx(G2, with_labels = True, node_size = 500)
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    xcord.insert(0,np.mean(xcord))
    ycord.insert(0,np.mean(ycord))

    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], distance_matrix[potential_nodes_p[r]][potential_nodes_p[h]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    #add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    
    
    
    distance_matrix_3=distance_matrix
   
    vertices = []

    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    
    priority=[1 for element in range(len(potential_nodes)+1)]
    updtd_prior = np.true_divide(priority, sum(priority))
    updated_priority_val = prior(G2, updtd_prior)
    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                    #add_edge(potential_nodes[h], potential_nodes[r], random.randint(20, 100))
    
    
    
    
    distance_matrix1=distance_matrix
    distance_matrix=distance_matrix_3
    
    
    initial_u1 = copy.deepcopy(potential_nodes)
    route=[]
    
    initial_nodes = copy.deepcopy(potential_nodes)
    initial_route_p = priority_dist(initial_u1, 0, distance_matrix1, G2,route)
            
    start_node_p = copy.deepcopy(start_node)
    p_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State_p = [[],[1],[],start_node_p]
    action_p = 0
    nodes_wp_p = []
    nodes_wop_visited_p = []
    nodes_wop_p = State_p[1] + State_p[3]
    cost_p=[]
    action_priority=[]
    nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
    for l in range(len(initial_route_p)):
        GG_p=[]
        if (initial_route_p[l] in nodes_visited_p):
            print("Priority Go to next node")
        else:
            if (initial_route_p[l] in nodes_notfaulty):
                
                cost_priority = (distance_matrix[action_p][initial_route_p[l]])*(len(initial_nodes) - len(nodes_wp_p))
            else:
                cost_priority = (distance_matrix[action_p][initial_route_p[l]] + service)*(len(initial_nodes) - len(nodes_wp_p))
            all_pre_p = list(nx.ancestors(G, initial_route_p[l]))
            if (all(item in nodes_wp_p for item in all_pre_p)):
                nodes_wp_p.append(initial_route_p[l])
            else:
                nodes_wop_visited_p.append(initial_route_p[l])
            
            action_p = initial_route_p[l]
            cost_p.append(cost_priority)
            action_priority.append(action_p)
            nodes_tobe_deleted_p =[]
            for k in range(len(p_nodes_notfaulty)):
                all_pre_p_check = list(nx.ancestors(G, p_nodes_notfaulty[k]))
                if (all(item in nodes_wp_p for item in all_pre_p_check)):
                    if (p_nodes_notfaulty[k] in nodes_wp_p):
                        nodes_wp_p = nodes_wp_p
                    else:
                        nodes_wp_p.append(p_nodes_notfaulty[k])
                    if (p_nodes_notfaulty[k] in nodes_wop_visited_p):
                        nodes_wop_visited_p.remove(p_nodes_notfaulty[k])
                    elif (p_nodes_notfaulty[k] in nodes_wop_p):
                        nodes_wop_p.remove(p_nodes_notfaulty[k])
                    nodes_tobe_deleted_p.append(p_nodes_notfaulty[k])
            p_nodes_notfaulty = [ele for ele in p_nodes_notfaulty if ele not in nodes_tobe_deleted_p]

            
            #extra_nodes_p = succ_prec(action_p, nodes_wop_visited_p,nodes_wp_p, p_nodes_notfaulty, GG_p)
            nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
            print("with power", nodes_wp_p)
            remaining_nodes_p =[ele for ele in initial_route_p if ele not in nodes_visited_p] 
    
    
    
    #print("Realization " + str(ast.literal_eval(realization[r1])))
    nodesnotfaulty = [ele for ele in potential_nodes if ele not in ast.literal_eval(realization[r1])]
    probval = pow(prob_value, len(ast.literal_eval(realization[r1]))) * pow((1- prob_value), len(nodesnotfaulty))
    
    #print("The optimal policy for priority algorithm is " + str(action_priority))
    #print("The cost for each action is " + str(cost_p))
    
    #print("The cost for priority algorithm is " + str(sum(cost_p)))
    
    
    #print("The optimal policy for nearest neighbour is " + str(action_near))
    #print("The cost for each action is " + str(cost_n))
    
    #print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    #print("The optimal policy is " + str(pol))
    #print("The cost without is " + str(cost))
    
    
    #print("The cost is " + str(sum(cost)))
    dynamic[0][0] = ast.literal_eval(realization[r1])
    dynamic[0][1] = probval
    dynamic[0][2] = action_priority
    dynamic[0][3] = sum(cost_p)
    dynamic[0][4] = action_near
    dynamic[0][5] = sum(cost_n)
    dynamic[0][6] = pol
    dynamic[0][7] = cost
    
    data_result = np.array(data_result, dtype=object)
    dynamic = np.array(dynamic, dtype=object)
    data_result = np.vstack([data_result, dynamic])
    
dat = pd.DataFrame(data_result, index=None)
dat.to_csv("40_test.csv", index=False, header=False)


prob_value = 0.75
service = 0.0




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

G = nx.DiGraph()

nodes = []
predecessor = []
xcord = []
ycord = []
weight =[]
  
filename = "Network100_Real.txt"
f = open(filename,'r')

with f as file:
    # Read the lines of the file
    lines = file.readlines()

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

for i,j in zip (nodes_updated,predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes=list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
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


nx.draw_networkx(G, with_labels = True, node_size = 100)




rows, cols = (1, 8)
data_result = [[0 for i in range(cols)] for j in range(rows)]


list_realization = pd.read_csv('Realization_0.75_Network100.csv')
realization = list(list_realization['realization'])       
    
for r1 in range(len(realization)):    
    print("Realization " + str(ast.literal_eval(realization[r1])))
    rows, cols = (1, 8)
    dynamic = [[0 for k in range(cols)] for j in range(rows)]
    cost = []
    start_node= copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in ast.literal_eval(realization[r1])]
    #nodes_notfaulty = [ele for ele in start_node if ele not in K]
    copy_nodes_notfaulty=copy.deepcopy(nodes_notfaulty)
    pol = 0
    cost = 0
            
        
    #print("The optimal policy is " + str(pol))
    #p#rint("The cost without is " + str(cost))
    
    #print("The cost is " + str(sum(cost)))
    
    ##############################################################################
    
    def nearest2(vec,k1,distance_matrix,G2, vec1):
        distance1=[]
        for j2 in range(len(vec)):   
            modified_distance = (distance_matrix[k1][vec[j2]])            
            distance1.append(modified_distance)
        #print(distance1)
        if (len(distance1)==0):
            res=[]
            temp = 0
        else:
            temp = min(distance1)
            #res = [i for i, j in enumerate(distance1) if j == temp]
            res = distance1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
    
        if (len(vec)!=0):
            nearest2(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    initial_u2 = potential_nodes
    initial_u1 = copy.deepcopy(initial_u2)
    route=[]
    #for i in range(len(potential_nodes)):
    #    if i != s2:
    #        initial_u1.append(i)
    

    initial_nodes = copy.deepcopy(initial_u1)
    
    initial_route = nearest2(initial_u1, 0, distance_matrix, G,route)
    
    

        
    start_node_n = copy.deepcopy(start_node)
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State = [[],[1],[],start_node_n]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n=[]
    action_near=[]
    nodes_visited = nodes_wp_n + nodes_wop_visited_n
    #remaining_nodes=copy.deepcopy(realization)
    for l in range(len(initial_route)):
        GG=[]
        if (initial_route[l] in nodes_visited):
            print("Go to next node")
        else:
            
            if (initial_route[l] in nodes_notfaulty):
                cost_near = (distance_matrix[action_n][initial_route[l]])*(len(initial_nodes) - len(nodes_wp_n))
            else:
                cost_near = (distance_matrix[action_n][initial_route[l]] + service)*(len(initial_nodes) - len(nodes_wp_n))
            all_pre_n = list(nx.ancestors(G, initial_route[l])) 
            if (all(item in nodes_wp_n for item in all_pre_n)):
                nodes_wp_n.append(initial_route[l])
            else:
                nodes_wop_visited_n.append(initial_route[l])
            
            action_n = initial_route[l]
            cost_n.append(cost_near)
            action_near.append(action_n)
            nodes_tobe_deleted_n =[]
            for k in range(len(N_nodes_notfaulty)):
                all_pre_n_check = list(nx.ancestors(G, N_nodes_notfaulty[k]))
                if (all(item in nodes_wp_n for item in all_pre_n_check)):
                    if (N_nodes_notfaulty[k] in nodes_wp_n):
                        nodes_wp_n = nodes_wp_n
                    else:
                        nodes_wp_n.append(N_nodes_notfaulty[k])
                    if (N_nodes_notfaulty[k] in nodes_wop_visited_n):
                        nodes_wop_visited_n.remove(N_nodes_notfaulty[k])
                    elif (N_nodes_notfaulty[k] in nodes_wop_n):
                        nodes_wop_n.remove(N_nodes_notfaulty[k])
                    nodes_tobe_deleted_n.append(N_nodes_notfaulty[k])
            N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in nodes_tobe_deleted_n]
        #    print(N_nodes_notfaulty, nodes_wop_visited_n,nodes_wp_n)
            #extra_nodes = succ_prec(action_n, nodes_wop_visited_n,nodes_wp_n,N_nodes_notfaulty, GG)
        #    print(extra_nodes)
            nodes_visited = nodes_wp_n + nodes_wop_visited_n
            remaining_nodes =[ele for ele in initial_route if ele not in nodes_visited]
        #    print(remaining_nodes)

    
    
    print("The optimal policy for nearest neighbour is " + str(action_near))
    print("The cost for each action is " + str(cost_n))
    
    print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    ##############################################################################
    
      
  
    def priority_dist(vec,k1,distance_matrix,G2, vec1):
        priority1=[]
        for j2 in range(len(vec)):   
            modified = (distance_matrix[k1][vec[j2]])            
            priority1.append(modified)
        #print(priority1)
        if (len(priority1)==0):
            res=[]
            temp = 0
        else:
            temp = max(priority1)
            res = priority1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
        if (len(vec)!=0):
            priority_dist(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    
    def prior(G2, pri):
        prio=[]
        for i in range(len(pri)):
            dec = list(nx.descendants(G2, i))
            prio.append(len(dec))
        #print(prio)
        updated_priority=prio*pri
        return updated_priority   
    
    vertices = []
    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    
    G2 = nx.DiGraph()
    
    nodes = []
    predecessor = []
    xcord = []
    ycord = []
    weight =[]
      
    f = open(filename,'r')

    with f as file:
        # Read the lines of the file
        lines = file.readlines()
    
    for line in lines:
        values = line.strip().split()
        nodes.append(int(values[0]))
        predecessor.append(int(values[1]))
        xcord.append(float(values[2]))
        ycord.append(float(values[3]))
        weight.append(int(values[4]))
        
    for i,j in zip (nodes,predecessor):
        G2.add_edges_from([(j, i)])
    
    nx.draw_networkx(G2, with_labels = True, node_size = 500)
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    xcord.insert(0,np.mean(xcord))
    ycord.insert(0,np.mean(ycord))

    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], distance_matrix[potential_nodes_p[r]][potential_nodes_p[h]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    #add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    
    
    
    distance_matrix_3=distance_matrix
   
    vertices = []

    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    
    priority=[1 for element in range(len(potential_nodes)+1)]
    updtd_prior = np.true_divide(priority, sum(priority))
    updated_priority_val = prior(G2, updtd_prior)
    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                    #add_edge(potential_nodes[h], potential_nodes[r], random.randint(20, 100))
    
    
    
    
    distance_matrix1=distance_matrix
    distance_matrix=distance_matrix_3
    
    
    initial_u1 = copy.deepcopy(potential_nodes)
    route=[]
    
    initial_nodes = copy.deepcopy(potential_nodes)
    initial_route_p = priority_dist(initial_u1, 0, distance_matrix1, G2,route)
            
    start_node_p = copy.deepcopy(start_node)
    p_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State_p = [[],[1],[],start_node_p]
    action_p = 0
    nodes_wp_p = []
    nodes_wop_visited_p = []
    nodes_wop_p = State_p[1] + State_p[3]
    cost_p=[]
    action_priority=[]
    nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
    for l in range(len(initial_route_p)):
        GG_p=[]
        if (initial_route_p[l] in nodes_visited_p):
            print("Priority Go to next node")
        else:
            if (initial_route_p[l] in nodes_notfaulty):
                
                cost_priority = (distance_matrix[action_p][initial_route_p[l]])*(len(initial_nodes) - len(nodes_wp_p))
            else:
                cost_priority = (distance_matrix[action_p][initial_route_p[l]] + service)*(len(initial_nodes) - len(nodes_wp_p))
            all_pre_p = list(nx.ancestors(G, initial_route_p[l]))
            if (all(item in nodes_wp_p for item in all_pre_p)):
                nodes_wp_p.append(initial_route_p[l])
            else:
                nodes_wop_visited_p.append(initial_route_p[l])
            
            action_p = initial_route_p[l]
            cost_p.append(cost_priority)
            action_priority.append(action_p)
            nodes_tobe_deleted_p =[]
            for k in range(len(p_nodes_notfaulty)):
                all_pre_p_check = list(nx.ancestors(G, p_nodes_notfaulty[k]))
                if (all(item in nodes_wp_p for item in all_pre_p_check)):
                    if (p_nodes_notfaulty[k] in nodes_wp_p):
                        nodes_wp_p = nodes_wp_p
                    else:
                        nodes_wp_p.append(p_nodes_notfaulty[k])
                    if (p_nodes_notfaulty[k] in nodes_wop_visited_p):
                        nodes_wop_visited_p.remove(p_nodes_notfaulty[k])
                    elif (p_nodes_notfaulty[k] in nodes_wop_p):
                        nodes_wop_p.remove(p_nodes_notfaulty[k])
                    nodes_tobe_deleted_p.append(p_nodes_notfaulty[k])
            p_nodes_notfaulty = [ele for ele in p_nodes_notfaulty if ele not in nodes_tobe_deleted_p]

            
            #extra_nodes_p = succ_prec(action_p, nodes_wop_visited_p,nodes_wp_p, p_nodes_notfaulty, GG_p)
            nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
            print("with power", nodes_wp_p)
            remaining_nodes_p =[ele for ele in initial_route_p if ele not in nodes_visited_p] 
    
    
    
    #print("Realization " + str(ast.literal_eval(realization[r1])))
    nodesnotfaulty = [ele for ele in potential_nodes if ele not in ast.literal_eval(realization[r1])]
    probval = pow(prob_value, len(ast.literal_eval(realization[r1]))) * pow((1- prob_value), len(nodesnotfaulty))
    
    #print("The optimal policy for priority algorithm is " + str(action_priority))
    #print("The cost for each action is " + str(cost_p))
    
    #print("The cost for priority algorithm is " + str(sum(cost_p)))
    
    
    #print("The optimal policy for nearest neighbour is " + str(action_near))
    #print("The cost for each action is " + str(cost_n))
    
    #print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    #print("The optimal policy is " + str(pol))
    #print("The cost without is " + str(cost))
    
    
    #print("The cost is " + str(sum(cost)))
    dynamic[0][0] = ast.literal_eval(realization[r1])
    dynamic[0][1] = probval
    dynamic[0][2] = action_priority
    dynamic[0][3] = sum(cost_p)
    dynamic[0][4] = action_near
    dynamic[0][5] = sum(cost_n)
    dynamic[0][6] = pol
    dynamic[0][7] = cost
    
    data_result = np.array(data_result, dtype=object)
    dynamic = np.array(dynamic, dtype=object)
    data_result = np.vstack([data_result, dynamic])
    
dat = pd.DataFrame(data_result, index=None)
dat.to_csv("40_test_2.csv", index=False, header=False)






prob_value = 0.90
service = 0.0




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

G = nx.DiGraph()

nodes = []
predecessor = []
xcord = []
ycord = []
weight =[]
  
filename = "Network100_Real.txt"
f = open(filename,'r')

with f as file:
    # Read the lines of the file
    lines = file.readlines()

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

for i,j in zip (nodes_updated,predecessor_update):
    G.add_edges_from([(j, i)])
potential_nodes=list(G.nodes())

update_potentialnodes = copy.deepcopy(potential_nodes)
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


nx.draw_networkx(G, with_labels = True, node_size = 100)




rows, cols = (1, 8)
data_result = [[0 for i in range(cols)] for j in range(rows)]


list_realization = pd.read_csv('Realization_0.90_Network100.csv')
realization = list(list_realization['realization'])       
    
for r1 in range(len(realization)):    
    print("Realization " + str(ast.literal_eval(realization[r1])))
    rows, cols = (1, 8)
    dynamic = [[0 for k in range(cols)] for j in range(rows)]
    cost = []
    start_node= copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in ast.literal_eval(realization[r1])]
    #nodes_notfaulty = [ele for ele in start_node if ele not in K]
    copy_nodes_notfaulty=copy.deepcopy(nodes_notfaulty)
    pol = 0
    cost = 0
            
        
    #print("The optimal policy is " + str(pol))
    #p#rint("The cost without is " + str(cost))
    
    #print("The cost is " + str(sum(cost)))
    
    ##############################################################################
    
    def nearest2(vec,k1,distance_matrix,G2, vec1):
        distance1=[]
        for j2 in range(len(vec)):   
            modified_distance = (distance_matrix[k1][vec[j2]])            
            distance1.append(modified_distance)
        #print(distance1)
        if (len(distance1)==0):
            res=[]
            temp = 0
        else:
            temp = min(distance1)
            #res = [i for i, j in enumerate(distance1) if j == temp]
            res = distance1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
    
        if (len(vec)!=0):
            nearest2(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    initial_u2 = potential_nodes
    initial_u1 = copy.deepcopy(initial_u2)
    route=[]
    #for i in range(len(potential_nodes)):
    #    if i != s2:
    #        initial_u1.append(i)
    

    initial_nodes = copy.deepcopy(initial_u1)
    
    initial_route = nearest2(initial_u1, 0, distance_matrix, G,route)
    
    

        
    start_node_n = copy.deepcopy(start_node)
    N_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State = [[],[1],[],start_node_n]
    action_n = 0
    nodes_wp_n = []
    nodes_wop_visited_n = []
    nodes_wop_n = State[1] + State[3]
    cost_n=[]
    action_near=[]
    nodes_visited = nodes_wp_n + nodes_wop_visited_n
    #remaining_nodes=copy.deepcopy(realization)
    for l in range(len(initial_route)):
        GG=[]
        if (initial_route[l] in nodes_visited):
            print("Go to next node")
        else:
            
            if (initial_route[l] in nodes_notfaulty):
                cost_near = (distance_matrix[action_n][initial_route[l]])*(len(initial_nodes) - len(nodes_wp_n))
            else:
                cost_near = (distance_matrix[action_n][initial_route[l]] + service)*(len(initial_nodes) - len(nodes_wp_n))
            all_pre_n = list(nx.ancestors(G, initial_route[l])) 
            if (all(item in nodes_wp_n for item in all_pre_n)):
                nodes_wp_n.append(initial_route[l])
            else:
                nodes_wop_visited_n.append(initial_route[l])
            
            action_n = initial_route[l]
            cost_n.append(cost_near)
            action_near.append(action_n)
            nodes_tobe_deleted_n =[]
            for k in range(len(N_nodes_notfaulty)):
                all_pre_n_check = list(nx.ancestors(G, N_nodes_notfaulty[k]))
                if (all(item in nodes_wp_n for item in all_pre_n_check)):
                    if (N_nodes_notfaulty[k] in nodes_wp_n):
                        nodes_wp_n = nodes_wp_n
                    else:
                        nodes_wp_n.append(N_nodes_notfaulty[k])
                    if (N_nodes_notfaulty[k] in nodes_wop_visited_n):
                        nodes_wop_visited_n.remove(N_nodes_notfaulty[k])
                    elif (N_nodes_notfaulty[k] in nodes_wop_n):
                        nodes_wop_n.remove(N_nodes_notfaulty[k])
                    nodes_tobe_deleted_n.append(N_nodes_notfaulty[k])
            N_nodes_notfaulty = [ele for ele in N_nodes_notfaulty if ele not in nodes_tobe_deleted_n]
        #    print(N_nodes_notfaulty, nodes_wop_visited_n,nodes_wp_n)
            #extra_nodes = succ_prec(action_n, nodes_wop_visited_n,nodes_wp_n,N_nodes_notfaulty, GG)
        #    print(extra_nodes)
            nodes_visited = nodes_wp_n + nodes_wop_visited_n
            remaining_nodes =[ele for ele in initial_route if ele not in nodes_visited]
        #    print(remaining_nodes)

    
    
    print("The optimal policy for nearest neighbour is " + str(action_near))
    print("The cost for each action is " + str(cost_n))
    
    print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    ##############################################################################
    
      
  
    def priority_dist(vec,k1,distance_matrix,G2, vec1):
        priority1=[]
        for j2 in range(len(vec)):   
            modified = (distance_matrix[k1][vec[j2]])            
            priority1.append(modified)
        #print(priority1)
        if (len(priority1)==0):
            res=[]
            temp = 0
        else:
            temp = max(priority1)
            res = priority1.index(temp)
        k1=vec[res]
        vec1.append(vec[res])
        vec.remove(vec[res])
        if (len(vec)!=0):
            priority_dist(vec, k1, distance_matrix, G2, vec1)
        return vec1
        
    
    
    def prior(G2, pri):
        prio=[]
        for i in range(len(pri)):
            dec = list(nx.descendants(G2, i))
            prio.append(len(dec))
        #print(prio)
        updated_priority=prio*pri
        return updated_priority   
    
    vertices = []
    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    
    G2 = nx.DiGraph()
    
    nodes = []
    predecessor = []
    xcord = []
    ycord = []
    weight =[]
      
    f = open(filename,'r')

    with f as file:
        # Read the lines of the file
        lines = file.readlines()
    
    for line in lines:
        values = line.strip().split()
        nodes.append(int(values[0]))
        predecessor.append(int(values[1]))
        xcord.append(float(values[2]))
        ycord.append(float(values[3]))
        weight.append(int(values[4]))
        
    for i,j in zip (nodes,predecessor):
        G2.add_edges_from([(j, i)])
    
    nx.draw_networkx(G2, with_labels = True, node_size = 500)
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    xcord.insert(0,np.mean(xcord))
    ycord.insert(0,np.mean(ycord))

    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], distance_matrix[potential_nodes_p[r]][potential_nodes_p[h]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    #add_edge(potential_nodes_p[h], potential_nodes_p[r], distance.euclidean((xcord[h],ycord[h]),(xcord[r],ycord[r])))
                    
    
    
    distance_matrix_3=distance_matrix
   
    vertices = []

    vertices_no = 0
    distance_matrix = []
    # Add vertices to the graph
    
    potential_nodes_p=list(G2.nodes())
    
    for h in range(len(potential_nodes_p)):
        add_vertex(potential_nodes_p[h])
    
    
    priority=[1 for element in range(len(potential_nodes)+1)]
    updtd_prior = np.true_divide(priority, sum(priority))
    updated_priority_val = prior(G2, updtd_prior)
    
    for h in range(len(potential_nodes_p)):
        for r in range(len(potential_nodes_p)):
            if (h ==r):
                add_edge(potential_nodes_p[h], potential_nodes_p[r], 0)
            else:
                if (potential_nodes_p[h]>potential_nodes_p[r] and h>0):
                    if ((check(potential_nodes_p[r], potential_nodes_p[h]))):
                        add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                else:
                    add_edge(potential_nodes_p[h], potential_nodes_p[r], updated_priority_val[potential_nodes_p[r]])
                    #add_edge(potential_nodes[h], potential_nodes[r], random.randint(20, 100))
    
    
    
    
    distance_matrix1=distance_matrix
    distance_matrix=distance_matrix_3
    
    
    initial_u1 = copy.deepcopy(potential_nodes)
    route=[]
    
    initial_nodes = copy.deepcopy(potential_nodes)
    initial_route_p = priority_dist(initial_u1, 0, distance_matrix1, G2,route)
            
    start_node_p = copy.deepcopy(start_node)
    p_nodes_notfaulty = copy.deepcopy(nodes_notfaulty)
    State_p = [[],[1],[],start_node_p]
    action_p = 0
    nodes_wp_p = []
    nodes_wop_visited_p = []
    nodes_wop_p = State_p[1] + State_p[3]
    cost_p=[]
    action_priority=[]
    nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
    for l in range(len(initial_route_p)):
        GG_p=[]
        if (initial_route_p[l] in nodes_visited_p):
            print("Priority Go to next node")
        else:
            if (initial_route_p[l] in nodes_notfaulty):
                
                cost_priority = (distance_matrix[action_p][initial_route_p[l]])*(len(initial_nodes) - len(nodes_wp_p))
            else:
                cost_priority = (distance_matrix[action_p][initial_route_p[l]] + service)*(len(initial_nodes) - len(nodes_wp_p))
            all_pre_p = list(nx.ancestors(G, initial_route_p[l]))
            if (all(item in nodes_wp_p for item in all_pre_p)):
                nodes_wp_p.append(initial_route_p[l])
            else:
                nodes_wop_visited_p.append(initial_route_p[l])
            
            action_p = initial_route_p[l]
            cost_p.append(cost_priority)
            action_priority.append(action_p)
            nodes_tobe_deleted_p =[]
            for k in range(len(p_nodes_notfaulty)):
                all_pre_p_check = list(nx.ancestors(G, p_nodes_notfaulty[k]))
                if (all(item in nodes_wp_p for item in all_pre_p_check)):
                    if (p_nodes_notfaulty[k] in nodes_wp_p):
                        nodes_wp_p = nodes_wp_p
                    else:
                        nodes_wp_p.append(p_nodes_notfaulty[k])
                    if (p_nodes_notfaulty[k] in nodes_wop_visited_p):
                        nodes_wop_visited_p.remove(p_nodes_notfaulty[k])
                    elif (p_nodes_notfaulty[k] in nodes_wop_p):
                        nodes_wop_p.remove(p_nodes_notfaulty[k])
                    nodes_tobe_deleted_p.append(p_nodes_notfaulty[k])
            p_nodes_notfaulty = [ele for ele in p_nodes_notfaulty if ele not in nodes_tobe_deleted_p]

            
            #extra_nodes_p = succ_prec(action_p, nodes_wop_visited_p,nodes_wp_p, p_nodes_notfaulty, GG_p)
            nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
            print("with power", nodes_wp_p)
            remaining_nodes_p =[ele for ele in initial_route_p if ele not in nodes_visited_p] 
    
    
    
    #print("Realization " + str(ast.literal_eval(realization[r1])))
    nodesnotfaulty = [ele for ele in potential_nodes if ele not in ast.literal_eval(realization[r1])]
    probval = pow(prob_value, len(ast.literal_eval(realization[r1]))) * pow((1- prob_value), len(nodesnotfaulty))
    
    #print("The optimal policy for priority algorithm is " + str(action_priority))
    #print("The cost for each action is " + str(cost_p))
    
    #print("The cost for priority algorithm is " + str(sum(cost_p)))
    
    
    #print("The optimal policy for nearest neighbour is " + str(action_near))
    #print("The cost for each action is " + str(cost_n))
    
    #print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    #print("The optimal policy is " + str(pol))
    #print("The cost without is " + str(cost))
    
    
    #print("The cost is " + str(sum(cost)))
    dynamic[0][0] = ast.literal_eval(realization[r1])
    dynamic[0][1] = probval
    dynamic[0][2] = action_priority
    dynamic[0][3] = sum(cost_p)
    dynamic[0][4] = action_near
    dynamic[0][5] = sum(cost_n)
    dynamic[0][6] = pol
    dynamic[0][7] = cost
    
    data_result = np.array(data_result, dtype=object)
    dynamic = np.array(dynamic, dtype=object)
    data_result = np.vstack([data_result, dynamic])
    
dat = pd.DataFrame(data_result, index=None)
dat.to_csv("40_test_3.csv", index=False, header=False)


