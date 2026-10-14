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

start = timeit.default_timer()

prob_value = float(sys.argv[1])
service = float(sys.argv[2])

epsilon =float(sys.argv[4])

iteration = int(sys.argv[5])


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
  

filename = (sys.argv[3])
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

#calculate the distance matrix
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



#function to find the simulation of information
def state_simuation(start,array_data, G, prob):
    nod = [suc for suc in G.successors(start)]

    length = len(nod)
    if (length > 1):
        for counter in range(len(nod)):            
            if (rn.uniform(0, 1) > prob):
                array_data.append(nod[counter])
                y = state_simuation(nod[counter], array_data, G, prob)
            else:
                y = array_data
            
    elif (length == 1):
        if (rn.uniform(0, 1) > prob):
            array_data.append(nod[0])
            y= state_simuation(nod[0], array_data, G, prob) 
        else:
            y = array_data
    else:
        y = array_data
    
    return y

def information(start, nodes_n_faulty, G, ar_dt):
    nod =[suc for suc in G.successors(start)]
    length = len(nod)
    if (length > 1):
        for counter in range(len(nod)):            
            if (nod[counter] in nodes_n_faulty):
                ar_dt.append(nod[counter])
                y = information(nod[counter], nodes_n_faulty, G, ar_dt)
            else:
                y = ar_dt
            
    elif (length == 1):
        if (nod[0] in nodes_n_faulty):
            ar_dt.append(nod[0])
            y= information(nod[0], nodes_n_faulty, G, ar_dt) 
        else:
            y = ar_dt
    else:
        y = ar_dt
    
    return y

  
def func_cal_SWOP1(nodes_wop, nodes_wp):
    nodes_swop_1 = []
    for i in range(len(nodes_wop)):
        pre = [pred for pred in G.predecessors(nodes_wop[i])]
        if (len(pre) == 0):
            nodes_swop_1.append(1)
        elif (pre[0] in nodes_wp):
            nodes_swop_1.append(nodes_wop[i])
    return nodes_swop_1    



def post_state_calculation(nodes_with_power, nodes_without_power, A1, random_node, info, NNF, G, probability):
    if (len(nodes_without_power)!=0):
        #print("a: " + str(random_node))
        post_decision_state=[]
        pre_decision_state=[]
        all_predecessors = list(nx.ancestors(G, random_node))
        if (any(item in all_predecessors for item in nodes_without_power)):
            
            A1.append(random_node)
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            nodes_without_power_1 = [ele for ele in nodes_without_power if ele not in A1]
            nodes_swop = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            
            nodes_with_power.sort()
            
            nodes_swop.sort()

            node_0 = [ele for ele in A1 if ele not in nodes_with_power]
            node_0.sort()

            nodes_1_0 = nodes_swop + A1
            nodes_p = [ele for ele in nodes_without_power if ele not in nodes_1_0]
            nodes_p.sort()
            nodes_swop.extend(nodes_p)
            nodes_swop.sort()
            nodes_p.clear()

            post_decision_state.append([random_node])
            post_decision_state.append(nodes_with_power)
            post_decision_state.append(nodes_swop)  
            post_decision_state.append(node_0)          
            post_decision_state.append(nodes_p)

            nodes_swop_pre = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            nodes_with_power.sort()
        
            nodes_swop_pre.sort()
            
            #post_decision_state.append(A1)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            
            nodes_1_0_pre = nodes_swop_pre + A1
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            nodes_swop_pre.extend(nodes_p_pre)
            nodes_swop_pre.sort()
            nodes_p_pre.clear()
            pre_decision_state.append(nodes_with_power)
            pre_decision_state.append(nodes_swop_pre)
            pre_decision_state.append(node_0_pre)
            pre_decision_state.append(nodes_p_pre)
            
        else:
            info = []
            
            nodes_swop_post = func_cal_SWOP1(nodes_without_power, nodes_with_power)
            nodes_swop_post.remove(random_node)
            update_nodes_power = nodes_with_power + [random_node]
            post_decision_state.append([random_node])
            update_nodes_power.sort()
            post_decision_state.append(update_nodes_power)
            nodes_swop_post.sort()
            post_decision_state.append(nodes_swop_post)
            node_0_post = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_post.sort()
            post_decision_state.append(node_0_post)
            nodes_1_0_post = nodes_swop_post + node_0_post
            nodes_p_post = [ele for ele in nodes_without_power if ele not in nodes_1_0_post]
            nodes_p_post.remove(random_node)
            nodes_p_post.sort()
            nodes_swop_post.extend(nodes_p_post)
            nodes_swop_post.sort()
            nodes_p_post.clear()


            post_decision_state.append([random_node])
            post_decision_state.append(update_nodes_power)
            post_decision_state.append(nodes_swop_post)
            post_decision_state.append(node_0_post)
            post_decision_state.append(nodes_p_post)

            NNF_func = NNF + A1
            nodes_with_power = information(random_node, NNF_func, G, info) 
            nodes_with_power = nodes_with_power + update_nodes_power            

            if (len(A1)==0):
                extra_information_revealed = nodes_with_power
                extra_information_revealed.sort()
            else:
                extra_information_revealed = [ele for ele in A1 if ele not in update_nodes_power]
                extra_information_revealed.sort()
            if (len(extra_information_revealed) == 0):
                nodes_with_power=update_nodes_power
            else:
                for h in range(len(extra_information_revealed)):
                    all_pred = list(nx.ancestors(G, extra_information_revealed[h]))
                    if (all(item in nodes_with_power for item in all_pred)):
                        nodes_with_power = nodes_with_power + [extra_information_revealed[h]]
                    else:
                        nodes_with_power = nodes_with_power
            nodes_with_power = list(set(nodes_with_power))
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            nodes_swop_pre = func_cal_SWOP1(nodes_without_power, nodes_with_power)
            nodes_with_power.sort()
            pre_decision_state.append(nodes_with_power)
            nodes_swop_pre.sort()
            pre_decision_state.append(nodes_swop_pre)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            pre_decision_state.append(node_0_pre)
            nodes_1_0_pre = nodes_swop_pre + node_0_pre
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            nodes_swop_pre.extend(nodes_p_pre)
            nodes_swop_pre.sort()
            nodes_p_pre.clear()

            pre_decision_state.append(nodes_with_power)
            pre_decision_state.append(nodes_swop_pre)
            pre_decision_state.append(node_0_pre)
            pre_decision_state.append(nodes_p_pre)


            A1 = A1 + nodes_with_power
            A1 = list(set(A1))

    else:
        exit

    return pre_decision_state, post_decision_state


def state_calculation_policy_training(nodes_with_power, nodes_without_power, A1, random_node, info, G, probability):
    if (len(nodes_without_power)!=0):
        post_decision_state=[]
        pre_decision_state=[]
        all_predecessors = list(nx.ancestors(G, random_node))
        if (any(item in all_predecessors for item in nodes_without_power)):
            
            A1.append(random_node)
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            nodes_without_power_1 = [ele for ele in nodes_without_power if ele not in A1]
            nodes_swop = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
    
            nodes_with_power.sort()
            
            nodes_swop.sort()
            
            #post_decision_state.append(A1)
            node_0 = [ele for ele in A1 if ele not in nodes_with_power]
            node_0.sort()
            
            nodes_1_0 = nodes_swop + A1
            nodes_p = [ele for ele in nodes_without_power if ele not in nodes_1_0]
            nodes_p.sort()
            nodes_swop.extend(nodes_p)
            nodes_swop.sort()
            nodes_p.clear()
            

            post_decision_state.append([random_node])
            post_decision_state.append(nodes_with_power)
            post_decision_state.append(nodes_swop)
            post_decision_state.append(node_0)
            post_decision_state.append(nodes_p)

            nodes_swop_pre = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            nodes_with_power.sort()
            
            nodes_swop_pre.sort()
            
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            
            nodes_1_0_pre = nodes_swop_pre + A1
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            nodes_swop_pre.extend(nodes_p_pre)
            nodes_swop_pre.sort()
            nodes_p_pre.clear()
            
        
            pre_decision_state.append(nodes_with_power)
            pre_decision_state.append(nodes_swop_pre)
            pre_decision_state.append(node_0_pre)
            pre_decision_state.append(nodes_p_pre)
            A1.remove(random_node)
            
        else:
            nodes_swop_post = func_cal_SWOP1(nodes_without_power, nodes_with_power)
            
            nodes_swop_post.remove(random_node)
            update_nodes_power = nodes_with_power + [random_node]

            update_nodes_power.sort()

            nodes_swop_post.sort()
            
            node_0_post = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_post.sort()
            
            nodes_1_0_post = nodes_swop_post + node_0_post
            nodes_p_post = [ele for ele in nodes_without_power if ele not in nodes_1_0_post]
            nodes_p_post.remove(random_node)
            nodes_p_post.sort()
            nodes_swop_post.extend(nodes_p_post)
            nodes_swop_post.sort()
            nodes_p_post.clear()

            

            post_decision_state.append([random_node])
            post_decision_state.append(update_nodes_power)
            post_decision_state.append(nodes_swop_post)
            post_decision_state.append(node_0_post)
            post_decision_state.append(nodes_p_post)

            nodes_with_power = nodes_with_power + update_nodes_power
            if (len(A1)==0):
                extra_information_revealed = nodes_with_power
                extra_information_revealed.sort()
            else:
                extra_information_revealed = [ele for ele in A1 if ele not in update_nodes_power]
                extra_information_revealed.sort()
            if (len(extra_information_revealed) == 0):
                nodes_with_power=update_nodes_power
            else:
                for h in range(len(extra_information_revealed)):
                    all_pred = list(nx.ancestors(G, extra_information_revealed[h]))
                    if (all(item in nodes_with_power for item in all_pred)):
                        nodes_with_power = nodes_with_power + [extra_information_revealed[h]]
                    else:
                        nodes_with_power = nodes_with_power
            nodes_with_power = list(set(nodes_with_power))
            #print("Nodes with power in decision epoch " + str(k) + " is " + str(nodes_with_power))
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            #print("Nodes without power are " + str(nodes_without_power))
            nodes_swop_pre = func_cal_SWOP1(nodes_without_power, nodes_with_power)
            nodes_with_power.sort()
            pre_decision_state.append(nodes_with_power)
            nodes_swop_pre.sort()
            pre_decision_state.append(nodes_swop_pre)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            pre_decision_state.append(node_0_pre)
            nodes_1_0_pre = nodes_swop_pre + node_0_pre
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            pre_decision_state.append(nodes_p_pre)
            A1 = A1 + nodes_with_power
            A1 = list(set(A1))

    else:
        exit
    return post_decision_state    
    

nx.draw_circular(G, with_labels = True, node_size = 500)

def compareList(l1,l2):
   l1.sort()
   l2.sort()
   if(l1==l2):
      return 1
   else:
      return 0

def compareList1(l1,l2):
   return int((set(l1)==set(l2)))


def common_member(a, b):
    result = [i for i in a if i in b]
    return result


class alpha_class: 
    def __init__(self, s, val): 
        self.s = s
        self.val = val
    def __hash__(self):
      return hash(self.s)
    
    def __eq__(self, other):
        return str(self.s) == str(other.s)


class state: 
    def __init__(self, s, val): 
        self.s = s
        self.val = val
    def __hash__(self):
      return hash(self.s)
    
    def __eq__(self, other):
        return str(self.s) == str(other.s)


def proposition_3_new(cur_nod_3, bit_node, S_P, S_1, G):
    S_P_H = S_P + S_1
    S_P_H.sort()
    l_3=[]
    if ((bit_node[1] == 0) and (bit_node[2] ==0 or bit_node[2] == 1)):
        k = cur_nod_3[1]
        i = cur_nod_3[0]
        j = cur_nod_3[2]
        centre_x = (xcord[i]+xcord[k])/2
        centre_y = (ycord[i]+ycord[k])/2
        rad = math.sqrt((xcord[k]-centre_x)**2 + (ycord[k]-centre_y)**2)
        for h1 in range(len(S_P_H)):
            if (((xcord[j] - centre_x)**2 + (ycord[j] - centre_y)**2 <= rad**2) and (distance_matrix[j][S_P_H[h1]] >= distance_matrix[k][S_P_H[h1]])):
                l_3.append(S_P_H[h1])
    return l_3 

    
min_new_dict={}    
new_dict={}
alpha_dict={}

start_node= copy.deepcopy(potential_nodes)
start_node.remove(1)

info = [] #array to get the information. Initially the information array is blank.
s1=[]
Initial_S = [[],[1],[],start_node] # Initial State
Int_S = [[0],[],[1],[],start_node]
Int_S[2].extend(Int_S[-1])   # add all elements of start_node to Int_S[2]
Int_S[2].sort()
Int_S[-1].clear() 
error=[]
#Int_S = [[4], [], [1], [2, 4], [3]]
actual_cost=[]
itr=[]

obj = state(tuple(map(tuple, Int_S)), 0)
new_dict[obj] = obj

alphaobj = alpha_class(tuple(map(tuple, Int_S)), 0)
alpha_dict[alphaobj] = alphaobj
minimum_value = 1000000 
start_iteration = 400000
end_iteration = iteration
step = 10000

check_array = list(range(start_iteration, end_iteration, step))
max_error_array = []
for i in range(1, iteration):
    if (i in check_array):
        previous_dict = {key: state(val.s, val.val) for key, val in new_dict.items()}
    print("===================ITERATION==================: " + str (i))
    temp_new_dict={}
    start_node_list= copy.deepcopy(potential_nodes)
    start_node_list.remove(1)
    start_list_realization=[]
    for j in range(len(start_node_list)):
        if (rn.uniform(0, 1) <= prob_value):
            start_list_realization.append(start_node_list[j])
    start_list_realization.insert(0,1)
    nodes_not_faulty_realization = list(set(potential_nodes)-set(start_list_realization))
    nodes_wp=[]
    nodes_wop_visited = []
    Initial_S = [[],[1],[],start_node]
    Initial_S[1].extend(Initial_S[-1])   # add all elements of start_node to Int_S[2]
    Initial_S[1].sort()
    Initial_S[-1].clear() 
    Final_S = [potential_nodes,[], [], []]
    nodes_wop_updated = Initial_S[1]
    n = 1
    action = 0
    #pos1=0
    Post=[[],[],[1],[],start_node]
    Post[1].extend(Post[-1])   # add all elements of start_node to Int_S[2]
    Post[-1].clear()
    s2=[]
    total = 0
    Int_S = [[0],[],[1],[],start_node]
    Int_S[2].extend(Int_S[-1])   # add all elements of start_node to Int_S[2]
    Int_S[2].sort()
    Int_S[-1].clear() 
    sub_nodes=[]
    nodes_wop=copy.deepcopy(nodes_wop_updated)
    action_sequence = []
    action_bit_sequence = []
    while (n <= len(potential_nodes)):
        #print("start " + str(Initial_S))
        nodes_wp=Initial_S[0]
        nodes_wop_visited = Initial_S[2]
        nodes_wop_1 = Initial_S[1]
        nodes_wop_p = Initial_S[3]
        #print("\033[4mDecision Epoch Exploitation\033[0m: " + str (n))
        def_value = 150000000

          
        for j in range(len(nodes_wop)):
            
            all_pred = list(nx.ancestors(G, nodes_wop[j]))
            
 
            if (all(item in Initial_S[0] for item in all_pred)):
            #if (nodes_wop[j] in nodes_wop_1):
                c = ((distance_matrix[action][nodes_wop[j]] + service)*(len(potential_nodes) - len(nodes_wp)))
            #if (nodes_wop[j] in nodes_wop_p):
            else:
                c = ((distance_matrix[action][nodes_wop[j]] + service)*(len(potential_nodes) - len(nodes_wp)))
            Post_pre = state_calculation_policy_training(nodes_wp, nodes_wop_updated, nodes_wop_visited, nodes_wop[j], info, G, prob_value)
            
            obj2 = state(tuple(map(tuple, Post_pre)), 0)
            if (obj2 in new_dict):
                value = c + new_dict[obj2].val
            else:
                value = c

                
            if (value < def_value):
                def_value = value
                counter = j
                #Q = prev_state
        action1=action
        action = nodes_wop[counter]

        bit = 1 if action in Initial_S[1] else -1
        if (len(action_sequence) < 3):
            action_sequence.append(action)
            action_bit_sequence.append(bit)
        else:
            action_sequence.append(action_sequence.pop(0))
            action_sequence[2]=action
            action_bit_sequence.append(action_bit_sequence.pop(0))
            action_bit_sequence[2]=bit


        #print("Action_taken = " + str(action))  
        if (action in nodes_not_faulty_realization):
            c_value = (distance_matrix[action1][action])*(len(potential_nodes) - len(nodes_wp))
        else:
            c_value = (distance_matrix[action1][action] + service)*(len(potential_nodes) - len(nodes_wp))
 
        Pre_S, Post_S = post_state_calculation(nodes_wp, nodes_wop_updated, nodes_wop_visited, action, info, nodes_not_faulty_realization, G, prob_value)
        obj2 = state(tuple(map(tuple, Post_S)), 0)
        if (obj2 in new_dict):
            def_value = c_value + new_dict[obj2].val
        else:
            def_value = c_value
        nodes_wp = Pre_S[0]
        nodes_wop_visited = Pre_S[2]
        nodes_wop_updated = Pre_S[1] + Pre_S[3]
        
        obj3 = state(tuple(map(tuple, Int_S)), 0)
        if (len(action_sequence) == 3):
            y = proposition_3_new(action_sequence, action_bit_sequence, Pre_S[3], Pre_S[1], G)
            if (nodes_wop_updated == y):
                sub_nodes_3=[]
                nodes_wop = list(set(nodes_wop_updated)-set(sub_nodes_3))
            else:
                sub_nodes_3=common_member(nodes_wop_updated, y)
                if (nodes_wop_updated == sub_nodes_3):
                    sub_nodes_3=[]
                    nodes_wop = list(set(nodes_wop_updated)-set(sub_nodes_3))
                else:
                    nodes_wop = list(set(nodes_wop_updated)-set(sub_nodes_3))
        elif (len(action_sequence) <3):
            nodes_wop = list(set(nodes_wop_updated))
        
        alphaobj2 = alpha_class(tuple(map(tuple, Int_S)), 0)
        if (alphaobj2 in alpha_dict):
            alpha_dict[alphaobj2].val = alpha_dict[alphaobj2].val + 1
            alpha = 1/pow((alpha_dict[alphaobj2].val),0.7)
        else:
            alphaobj_new = alpha_class(tuple(map(tuple, Int_S)), 1)
            alpha_dict[alphaobj_new] = alphaobj_new
            alpha = 1/pow((alpha_dict[alphaobj_new].val),0.7)
        if (obj3 in new_dict):
            new_value = (1 - alpha)*new_dict[obj3].val + alpha*(def_value)
            new_dict[obj3].val = new_value
            obj3_temp = state(tuple(map(tuple, Int_S)), new_value)
            temp_new_dict[obj3_temp]=obj3_temp
        else:
            new_value = (1 - alpha)*0 + alpha*(def_value)
            obj_new = state(tuple(map(tuple, Int_S)), new_value)
            new_dict[obj_new] = obj_new
            temp_new_dict[obj_new]=obj_new
            
        #print ("******************************")
        if (compareList(Pre_S[0],Final_S[0]) == 1):
            break
        n += 1
        Int_S = Post_S
        Initial_S = Pre_S
    if ((i in check_array) and len(new_dict) ==len(previous_dict)):
        total_error = []  # Initialize the total error across all values
        selected_keys = {key: value_A for key, value_A in new_dict.items() if key in alpha_dict and alpha_dict[key].val > 5000}

        for key, obj in selected_keys.items():
            prev_obj = previous_dict[key]
            error = abs(obj.val - prev_obj.val)
            #total_error += error
            total_error.append(error)
            #updated_dict[key] = state(obj.s, obj.val)
            print("Loop working")


        max_error = max(total_error)
        max_error_array.append(max_error)
        previous_dict = {key: state(val.s, val.val) for key, val in new_dict.items()}
    if (len(max_error_array) > 5):        
        if (max(max_error_array[-5:]) <= epsilon):
            print("Threshold met. Exiting the iteration loop.")
            break  # Exit the loop if the error is below the threshold

        # Update the previous_dict for the next iteration
        #previous_dict = {key: state(val.s, val.val) for key, val in new_dict.items()}



num_of_itr = i
df = pd.DataFrame([t.__dict__ for t in new_dict ])
s = str(service)
p = str(prob_value)
#df.to_csv(s + "_States_"+ filename +"_" + p +".csv")

end = timeit.default_timer()
total_time = (end - start)


start1 = timeit.default_timer()

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
  
filename = (sys.argv[3])
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


nx.draw_circular(G, with_labels = True, node_size = 500)

def compareList(l1,l2):
   l1.sort()
   l2.sort()
   if(l1==l2):
      return 1
   else:
      return 0

def Average(lst):
    return sum(lst) / len(lst)

def state_calculation_policy(nodes_with_power, nodes_without_power, A1, random_node, info, G, probability):
    if (len(nodes_without_power)!=0):
        #print("a: " + str(random_node))
        post_decision_state=[]
        pre_decision_state=[]
        #nodes_without_power = [ele for ele in initial_nodes if ele not in nodes_with_power]
        all_predecessors = list(nx.ancestors(G, random_node))
        #print("Predecesors are " + str (all_predecessors))nodes_without_power=[]
        if (any(item in all_predecessors for item in nodes_without_power)):
            
            A1.append(random_node)
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            #nodes_without_power_1 = [ele for ele in list(G.nodes)if ele not in nodes_without_power_but_visited]
            nodes_without_power_1 = [ele for ele in nodes_without_power if ele not in A1]
            nodes_swop = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            
            nodes_with_power.sort()
            
            nodes_swop.sort()
            post_decision_state.append(nodes_swop)
            #post_decision_state.append(A1)
            node_0 = [ele for ele in A1 if ele not in nodes_with_power]
            node_0.sort()
            
            nodes_1_0 = nodes_swop + A1
            nodes_p = [ele for ele in nodes_without_power if ele not in nodes_1_0]
            nodes_p.sort()
            nodes_swop.extend(nodes_p)
            nodes_swop.sort()
            nodes_p.clear()
            
            post_decision_state.append([random_node])
            post_decision_state.append(nodes_with_power)
            post_decision_state.append(nodes_swop)
            post_decision_state.append(node_0)
            post_decision_state.append(nodes_p)

            nodes_swop_pre = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            nodes_with_power.sort()
            
            nodes_swop_pre.sort()
            
            #post_decision_state.append(A1)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            
            nodes_1_0_pre = nodes_swop_pre + A1
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            nodes_swop_pre.extend(nodes_p_pre)
            nodes_swop_pre.sort()
            nodes_p_pre.clear
            
            pre_decision_state.append(nodes_with_power)
            pre_decision_state.append(nodes_swop_pre)
            pre_decision_state.append(node_0_pre)
            pre_decision_state.append(nodes_p_pre)
            A1.remove(random_node)
           
        else:
            nodes_swop_post = func_cal_SWOP1(nodes_without_power, nodes_with_power)
            if (len(nodes_swop_post) ==0):
                print("No nodes to delete")
            else:
                if (random_node in nodes_swop_post):
                    nodes_swop_post.remove(random_node)
            #nodes_swop_post.remove(random_node)
            update_nodes_power = nodes_with_power + [random_node]
            
            update_nodes_power.sort()
            
            nodes_swop_post.sort()
            
            node_0_post = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_post.sort()
            
            nodes_1_0_post = nodes_swop_post + node_0_post
            nodes_p_post = [ele for ele in nodes_without_power if ele not in nodes_1_0_post]
            nodes_p_post.remove(random_node)
            nodes_p_post.sort()

            nodes_swop_post.extend(nodes_p_post)
            nodes_swop_post.sort()
            nodes_p_post.clear()
            
            post_decision_state.append([random_node])
            post_decision_state.append(update_nodes_power)
            post_decision_state.append(nodes_swop_post)
            post_decision_state.append(node_0_post)
            post_decision_state.append(nodes_p_post)


            post_decision_state.append(nodes_p_post)
            nodes_with_power = nodes_with_power + update_nodes_power
            if (len(A1)==0):
                extra_information_revealed = nodes_with_power
                extra_information_revealed.sort()
            else:
                extra_information_revealed = [ele for ele in A1 if ele not in update_nodes_power]
                extra_information_revealed.sort()
            if (len(extra_information_revealed) == 0):
                nodes_with_power=update_nodes_power
            else:
                for h in range(len(extra_information_revealed)):
                    all_pred = list(nx.ancestors(G, extra_information_revealed[h]))
                    if (all(item in nodes_with_power for item in all_pred)):
                        nodes_with_power = nodes_with_power + [extra_information_revealed[h]]
                    else:
                        nodes_with_power = nodes_with_power
            nodes_with_power = list(set(nodes_with_power))
            #print("Nodes with power in decision epoch " + str(k) + " is " + str(nodes_with_power))
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            #print("Nodes without power are " + str(nodes_without_power))
            nodes_swop_pre = func_cal_SWOP1(nodes_without_power, nodes_with_power)
            nodes_with_power.sort()
            pre_decision_state.append(nodes_with_power)
            nodes_swop_pre.sort()
            pre_decision_state.append(nodes_swop_pre)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            pre_decision_state.append(node_0_pre)
            nodes_1_0_pre = nodes_swop_pre + node_0_pre
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            nodes_swop_pre.extend(nodes_p_pre)
            nodes_swop_pre.sort()
            nodes_p_pre.clear()

            pre_decision_state.append(nodes_with_power)
            pre_decision_state.append(nodes_swop_pre)
            pre_decision_state.append(node_0_pre)
            pre_decision_state.append(nodes_p_pre)
            A1 = A1 + nodes_with_power
            A1 = list(set(A1))
            nodes_without_power_2 = nodes_p_pre + nodes_swop_pre
            if (len(nodes_without_power_2)==0):
                print("All the nodes have gained power")
    else:
        exit

    return pre_decision_state, post_decision_state

def succ_prec(current_node, wop_0, wp, wp_start, info_reveal):
    pre = list(nx.ancestors(G, current_node))
    wop_0 = wop_0 + wp_start
    if (len(list(G.successors(current_node))) != 0):        
        if ((list(G.successors(current_node))[0] in wop_0) and (all(item in wp for item in pre))):
            info_reveal.append(list(G.successors(current_node))[0])
            wp.append(list(G.successors(current_node))[0])
            wop_0.remove(list(G.successors(current_node))[0])
            current_node = list(G.successors(current_node))[0]
            succ_prec (current_node, wop_0, wp, wp_start, info_reveal)
        else:
            print("no one gained power")
    return info_reveal

rows, cols = (1, 8)
data_result = [[0 for i in range(cols)] for j in range(rows)]


list_realization = pd.read_csv('Realization_0.90_Network20.csv')
realization = list(list_realization['realization'])       
    
for r1 in range(len(realization)):    
    rows, cols = (1, 8)
    dynamic = [[0 for k in range(cols)] for j in range(rows)]
    cost = []
    start_node= copy.deepcopy(potential_nodes)
    start_node.remove(1)
    nodes_notfaulty = [ele for ele in start_node if ele not in ast.literal_eval(realization[r1])]
    copy_nodes_notfaulty=copy.deepcopy(nodes_notfaulty)
    realization_adp = copy.deepcopy(ast.literal_eval(realization[r1]))
    realization_adp.remove(1)
    info = []
    s1=[]
    s2=[]
    rows, cols = (1, 1)
    value_fn = [[0 for i in range(cols)] for j in range(rows)]
    
    Initial_S = [[],[1], [], start_node]
    Initial_S[1].extend(Initial_S[-1])   # add all elements of start_node to Int_S[2]
    Initial_S[1].sort()
    Initial_S[-1].clear()
    value_fn[0][0] = Initial_S
      
    
    nodes_wp=[]
    nodes_wop_visited = []
    Final_S = [potential_nodes,[], [], []]
    nodes_wop = Initial_S[1] + Initial_S[3]
    nodes_wop_1 = Initial_S[1]
    n = 1
    action = 0
    pos1=0
    pol=[]
    #Post_S=[[0],[],[1], nodes_notfaulty, realization_adp]
    while (n <= len(potential_nodes)):
        #print("\033[4mDecision Epoch\033[0m: " + str (n))
        def_value = 15000000
        val = []
        a = []
        tot_val = 0
        print("Nodes with out power" + str(nodes_wop))
        if (len(nodes_wop)==0):
            break;
        else:
            for j in range(len(nodes_wop)):
                track = 0
                Pre, Post = state_calculation_policy(nodes_wp, nodes_wop, nodes_wop_visited, nodes_wop[j], info, G, prob_value)
                #print(Pre)
                #print(Post)
                if (nodes_wop[j] in nodes_wop_1):
                    immediatecost = ((distance_matrix[action][nodes_wop[j]] + service)*(len(potential_nodes) - len(nodes_wp)))
                else:
                    immediatecost = ((distance_matrix[action][nodes_wop[j]] + service)*(len(potential_nodes) - len(nodes_wp)))
                
                obj_val = state(tuple(map(tuple, Post)), 0)
                if (obj_val in new_dict):
                #    print("Match found")
                    tot_val= immediatecost + new_dict[obj_val].val
                    val.append(tot_val)
                    a.append(nodes_wop[j])

                if (tot_val < def_value):
                    def_value = tot_val
                    counter = j
                    Pre_S = Pre

        if (nodes_wop[counter] in nodes_notfaulty):
            c_val = (distance_matrix[action][nodes_wop[counter]])*(len(potential_nodes) - len(nodes_wp))
        else:
            c_val = (distance_matrix[action][nodes_wop[counter]] + service)*(len(potential_nodes) - len(nodes_wp))
        if (len(a)==0):
            arr_cost = []
            for act in range(len(nodes_wop)):
                if (nodes_wop[counter] in nodes_notfaulty):
                    cost_val = (distance_matrix[action][nodes_wop[act]])*(len(potential_nodes) - len(nodes_wp))
                else:
                    cost_val = (distance_matrix[action][nodes_wop[act]] + service)*(len(potential_nodes) - len(nodes_wp))
                arr_cost.append(cost_val)
            index_count=np.argmin(arr_cost)
            c_val = min(arr_cost)
            Pre, Post = state_calculation_policy(nodes_wp, nodes_wop, nodes_wop_visited, nodes_wop[index_count], info, G, prob_value)
            Pre_S = Pre
            obj_val = state(tuple(map(tuple, Post)), 0)
            counter = index_count

        
        action = nodes_wop[counter]
        #print(Pre_S)
        print("The action is " + str(action))
        print("Value is " + str(c_val))
        pol.append(action)
        #cost.append(def_value)
        cost.append(c_val)
        nodes_tobe_deleted = []
        nodes_wp = Pre_S[0]
        nodes_wop_visited = Pre_S[2]
        nodes_wop = Pre_S[1] + Pre_S[3]
        nodes_wop_1 = Pre_S[1]
        copy_nodes_notfaulty = copy_nodes_notfaulty + nodes_wop_visited
        copy_nodes_notfaulty.sort()
        #print("ggkk " + str(copy_nodes_notfaulty))
        for k in range(len(copy_nodes_notfaulty)):
            all_pre = list(nx.ancestors(G, copy_nodes_notfaulty[k]))
            if (all(item in nodes_wp for item in all_pre)):
                if (copy_nodes_notfaulty[k] in nodes_wp):
                    nodes_wp = nodes_wp
                else:
                    nodes_wp.append(copy_nodes_notfaulty[k])
                if (copy_nodes_notfaulty[k] in nodes_wop_visited):
                    nodes_wop_visited.remove(copy_nodes_notfaulty[k])
                elif (copy_nodes_notfaulty[k] in nodes_wop):
                    nodes_wop.remove(copy_nodes_notfaulty[k])
                nodes_tobe_deleted.append(copy_nodes_notfaulty[k])
        temp_nodes_wop_1=[]
        for wp1 in range(len(nodes_wp)):
            succ_wp1 = [suc for suc in G.successors(nodes_wp[wp1])]
            temp_not_pow = list(set(succ_wp1) - set(nodes_wp))
            temp_nodes_wop_1.append(temp_not_pow)
        nodes_wop_1 = [item for sublist in temp_nodes_wop_1 for item in sublist]    
            
        copy_nodes_notfaulty = [ele for ele in copy_nodes_notfaulty if ele not in nodes_tobe_deleted]
              
        
       
        if (compareList(Pre_S[0],Final_S[0]) == 1):
            break
        
        n += 1
        
        
    print("The optimal policy is " + str(pol))
    print("The cost without is " + str(cost))
    
    print("The cost is " + str(sum(cost)))
    
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
    
    realization_nn = copy.deepcopy(ast.literal_eval(realization[r1]))
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
            extra_nodes = succ_prec(action_n, nodes_wop_visited_n,nodes_wp_n,N_nodes_notfaulty, GG)
        #    print(extra_nodes)
            nodes_visited = nodes_wp_n + nodes_wop_visited_n
            remaining_nodes =[ele for ele in initial_route if ele not in nodes_visited]
        #    print(remaining_nodes)

    
    
    print("The optimal policy for nearest neighbour is " + str(action_near))
    print("The cost for each action is " + str(cost_n))
    
    print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    ##############################################################################
    
      
    
    def nearest1(vec,k1,distance_matrix,G2, vec1):
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
            nearest1(vec, k1, distance_matrix, G2, vec1)
        return vec1
    
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
      
    filename = (sys.argv[3])
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
    
    nx.draw(G2, with_labels = True, node_size = 500)
    
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
            print("Go to next node")
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
            extra_nodes_p = succ_prec(action_p, nodes_wop_visited_p,nodes_wp_p, p_nodes_notfaulty, GG_p)
            nodes_visited_p = nodes_wp_p + nodes_wop_visited_p
            remaining_nodes_p =[ele for ele in initial_route_p if ele not in nodes_visited_p] 
    
    
    
    print("Realization " + str(ast.literal_eval(realization[r1])))
    nodesnotfaulty = [ele for ele in potential_nodes if ele not in ast.literal_eval(realization[r1])]
    probval = pow(prob_value, len(ast.literal_eval(realization[r1]))) * pow((1- prob_value), len(nodesnotfaulty))
    
    print("The optimal policy for priority algorithm is " + str(action_priority))
    print("The cost for each action is " + str(cost_p))
    
    print("The cost for priority algorithm is " + str(sum(cost_p)))
    
    
    print("The optimal policy for nearest neighbour is " + str(action_near))
    print("The cost for each action is " + str(cost_n))
    
    print("The cost for nearest neighbour is " + str(sum(cost_n)))
    
    
    print("The optimal policy is " + str(pol))
    print("The cost without is " + str(cost))
    
    
    print("The cost is " + str(sum(cost)))
    dynamic[0][0] = ast.literal_eval(realization[r1])
    dynamic[0][1] = probval
    dynamic[0][2] = action_priority
    dynamic[0][3] = sum(cost_p)
    dynamic[0][4] = action_near
    dynamic[0][5] = sum(cost_n)
    dynamic[0][6] = pol
    dynamic[0][7] = sum(cost)
    
    
    data_result = np.vstack([data_result, dynamic])
    
dat = pd.DataFrame(data_result, index=None)
column_names = {0:'Realization', 1:'prob', 2:'Action_P', 3:'P', 4:'Action_NN', 5:'NN', 6:'Action_SM', 7:'SM'}
dat.rename(columns=column_names, inplace=True)
dat = dat.drop(0)
dat.reset_index(drop=True, inplace=True)
print(dat)
dat['P/SM'] = dat['P'] / dat['SM']
dat['NN/SM'] = dat['NN']/dat['SM']


s = str(service)
p = str(prob_value)
#dat.to_excel(s + "_Table_" + filename + "_" + p +".xlsx")
dat.to_csv(s + "_Table_" + filename + "_" + p +".csv", index=False)

#end = time.time()
#total_time = end - start
end1 = timeit.default_timer()
total_time1 = (end1 - start1)
print("\n"+ str(total_time1))

columns_to_average = ['SM', 'P', 'NN', 'P/SM', 'NN/SM']
average_array = dat[columns_to_average].mean().values
average_dat = pd.DataFrame([average_array], columns=columns_to_average, index=None)
average_dat.insert(0, 'p_value', p)
average_dat.insert(0, 's', s)
average_dat['States'] = df.shape[0]
average_dat['t1(sec)'] = total_time
average_dat['iteration'] = num_of_itr
average_dat['t2(sec)'] = total_time1

s = str(service)
p = str(prob_value)
#dat.to_excel(s + "_Table_" + filename + "_" + p +".xlsx")
average_dat.to_csv(s + "Computational_Table_" + filename + "_" + p +".csv", index=False)
print("\n" + str(num_of_itr))


