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
#import ast
#start = time.time()
prob_value = 0.90
service = 0.0
#prob_value = float(sys.argv[1])
#service = float(sys.argv[2])
epsilon = 0.05
#epsilon =float(sys.argv[4])
iteration = 998
#iteration = int(sys.argv[5])
f = open('Network26_OM.txt','r')


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
            post_decision_state.append([random_node])
            nodes_with_power.sort()
            post_decision_state.append(nodes_with_power)
            nodes_swop.sort()
            post_decision_state.append(nodes_swop)
            node_0 = [ele for ele in A1 if ele not in nodes_with_power]
            node_0.sort()
            post_decision_state.append(node_0)
            nodes_1_0 = nodes_swop + A1
            nodes_p = [ele for ele in nodes_without_power if ele not in nodes_1_0]
            nodes_p.sort()
            post_decision_state.append(nodes_p)
            nodes_swop_pre = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            nodes_with_power.sort()
            pre_decision_state.append(nodes_with_power)
            nodes_swop_pre.sort()
            pre_decision_state.append(nodes_swop_pre)
            #post_decision_state.append(A1)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            pre_decision_state.append(node_0_pre)
            nodes_1_0_pre = nodes_swop_pre + A1
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
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
            pre_decision_state.append(nodes_p_pre)
            A1 = A1 + nodes_with_power
            A1 = list(set(A1))

    else:
        exit

    return pre_decision_state, post_decision_state


def state_calculation_policy(nodes_with_power, nodes_without_power, A1, random_node, info, G, probability):
    if (len(nodes_without_power)!=0):
        post_decision_state=[]
        pre_decision_state=[]
        all_predecessors = list(nx.ancestors(G, random_node))
        if (any(item in all_predecessors for item in nodes_without_power)):
            
            A1.append(random_node)
            nodes_without_power = [ele for ele in list(G.nodes) if ele not in nodes_with_power]
            nodes_without_power_1 = [ele for ele in nodes_without_power if ele not in A1]
            nodes_swop = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            post_decision_state.append([random_node])
            nodes_with_power.sort()
            post_decision_state.append(nodes_with_power)
            nodes_swop.sort()
            post_decision_state.append(nodes_swop)
            #post_decision_state.append(A1)
            node_0 = [ele for ele in A1 if ele not in nodes_with_power]
            node_0.sort()
            post_decision_state.append(node_0)
            nodes_1_0 = nodes_swop + A1
            nodes_p = [ele for ele in nodes_without_power if ele not in nodes_1_0]
            nodes_p.sort()
            post_decision_state.append(nodes_p)
            nodes_swop_pre = func_cal_SWOP1(nodes_without_power_1, nodes_with_power)
            nodes_with_power.sort()
            pre_decision_state.append(nodes_with_power)
            nodes_swop_pre.sort()
            pre_decision_state.append(nodes_swop_pre)
            node_0_pre = [ele for ele in A1 if ele not in nodes_with_power]
            node_0_pre.sort()
            pre_decision_state.append(node_0_pre)
            nodes_1_0_pre = nodes_swop_pre + A1
            nodes_p_pre = [ele for ele in nodes_without_power if ele not in nodes_1_0_pre]
            nodes_p_pre.sort()
            pre_decision_state.append(nodes_p_pre)
            A1.remove(random_node)
            
        else:
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
    

print_graph()

nx.draw_networkx(G, with_labels = True, node_size = 500)

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


  
min_new_dict={}    
new_dict={}
alpha_dict={}

hopeless = pd.read_csv('Hopeless.csv')
#hopeless1['col'] = hopeless1['col'].apply(lambda x: list(map(int, x.split(','))))

hopeless['col'] = hopeless['col'].apply(ast.literal_eval)

state_OM = pd.read_csv('State_OM.csv')
#hopeless1['col'] = hopeless1['col'].apply(lambda x: list(map(int, x.split(','))))

state_OM['state'] = state_OM['state'].apply(ast.literal_eval)
OM_list = state_OM["state"].tolist()


start_node= copy.deepcopy(potential_nodes)
start_node.remove(1)
info = [] #array to get the information. Initially the information array is blank.
s1=[]
Initial_S = [[],[1],[],start_node] # Initial State
Int_S = [[0],[],[1],[],start_node]
error=[]
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
state_data_OM = {}

check_array = list(range(start_iteration, end_iteration, step))
max_error_array = []
Time=[]
for i in range(1, iteration):
    if (i in check_array):
        previous_dict = {key: state(val.s, val.val) for key, val in new_dict.items()}

    print("===================ITERATION==================: " + str (i))
    temp_new_dict={}
    start_node_list= copy.deepcopy(potential_nodes)
    start_node_list.remove(1)
    start_list_realization=hopeless['col'][i]    
    #start_list_realization=hopeless['col'][i]
    #print("Realization: " + str(start_list_realization))
    nodes_not_faulty_realization = list(set(potential_nodes)-set(start_list_realization))
    
    nodes_wp=[]
    nodes_wop_visited = []
    Initial_S = [[],[1],[],start_node]
    Final_S = [potential_nodes,[], [], []]
    nodes_wop_updated = Initial_S[1] + Initial_S[3]
    n = 1
    action = 0
    #pos1=0
    Post=[[],[],[1],[],start_node]
    s2=[]
    total = 0
    Int_S = [[0],[],[1],[],start_node]

    nodes_wop=copy.deepcopy(nodes_wop_updated)
    
    while (n <= len(potential_nodes)):
        #print("start " + str(Initial_S))
        start_i = timeit.default_timer()
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
                c = (((distance_matrix[action][nodes_wop[j]] + service)*(len(potential_nodes) - len(nodes_wp)))*prob_value + ((distance_matrix[action][nodes_wop[j]])*(len(potential_nodes) - len(nodes_wp)))*(1 - prob_value))
            Post_pre = state_calculation_policy(nodes_wp, nodes_wop_updated, nodes_wop_visited, nodes_wop[j], info, G, prob_value)
            
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
        
        

        if (tuple(tuple(sub) for sub in Post_S) in OM_list):
            state_name = tuple(tuple(sub) for sub in Post_S)  # Replace with your method to generate state names
            value = def_value  # Replace with your method to generate values
            
            # Check if the state is already in the dictionary
            if state_name in state_data_OM:
                state_data_OM[state_name].append(value)
            else:
                state_data_OM[state_name] = [value]

        
        nodes_wp = Pre_S[0]
        nodes_wop_visited = Pre_S[2]
        nodes_wop_updated = Pre_S[1] + Pre_S[3]
        
        obj3 = state(tuple(map(tuple, Int_S)), 0)
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
        
        #pos1 = pos
        n += 1
        Int_S = Post_S
        Initial_S = Pre_S
        end_i = timeit.default_timer()
        total_time_i = (end_i - start_i)
        Time.append([total_time_i,i,n])    
    if ((i in check_array) and len(new_dict) ==len(previous_dict)):
        total_error = []  # Initialize the total error across all values

        for key, obj in new_dict.items():
            prev_obj = previous_dict[key]
            error = abs(obj.val - prev_obj.val)
            #total_error += error
            total_error.append(error)
            #updated_dict[key] = state(obj.s, obj.val)

        max_error = max(total_error)
        max_error_array.append(max_error)
        previous_dict = {key: state(val.s, val.val) for key, val in new_dict.items()}
    if (len(max_error_array) > 20):        
        if (max(max_error_array[-20:]) <= epsilon):
            print("Threshold met. Exiting the iteration loop.")
            break  # Exit the loop if the error is below the threshold

num_of_itr = i
df = pd.DataFrame([t.__dict__ for t in new_dict ])
Time_df=pd.DataFrame(Time)
Time_df.to_csv("Time_OM.csv")
s = str(service)
p = str(prob_value)
#df.to_csv(s + "_States_"+ filename +"_" + p +".csv")



max_length = max(len(lst) for lst in state_data_OM.values())

# Fill in missing values in the lists with NaN
for key, value in state_data_OM.items():
    if len(value) < max_length:
        state_data_OM[key] += [None] * (max_length - len(value))

# Convert the processed dictionary to a DataFrame

mapping_df = pd.read_csv('key_mapping.csv')
mapping_df['state'] = mapping_df['state'].apply(ast.literal_eval)

# Convert the DataFrame to a dictionary with 'old_key' as keys and 'new_key' as values
key_mapping = dict(zip(mapping_df['state'], mapping_df['location']))
state_data_OM_updated = {key_mapping.get(key, key): value for key, value in state_data_OM.items()}


state_df_OM = pd.DataFrame(state_data_OM_updated)

state_df_OM.to_csv("OM.csv")


OM_State = pd.read_csv("0.0_OM_Network16_OM.txt_0.5.csv")
SM_State = pd.read_csv("0.0_SM_Network16_SM.txt_0.5.csv")

OM_State = OM_State.drop(columns=['Unnamed: 0'])
SM_State = SM_State.drop(columns=['Unnamed: 0'])

count_OM = OM_State.shape[1]
count_SM = SM_State.shape[1]

state=[]
SM=[]
OM=[]

for i in range(1, count_OM+1):
    state.append(i)
    f_SM = SM_State[str(i)]
    f_OM = OM_State[str(i)]
    state_6_SM = f_SM.dropna().to_numpy()
    
    last_value_SM = state_6_SM[-1]
    
    # Divide every element in the array by the last value
    result_array_SM = [x / last_value_SM for x in state_6_SM]
    res_SM = np.array(result_array_SM)
    threshold = 1.0  # The value you want to find
    
    # Find the index when the threshold is crossed using NumPy
    indices_SM = np.where(res_SM >= threshold)
    
    if indices_SM[0].size > 0:
        first_crossing_index_SM = indices_SM[0][0]
        print(f"The threshold {threshold} is first crossed at index {first_crossing_index_SM}.")
    
    
    SM.append(first_crossing_index_SM)
    values_to_store_SM = res_SM[:first_crossing_index_SM].copy()
    
    
    state_6_OM = f_OM.dropna().to_numpy()
    
    last_value_OM = state_6_OM[-1]
    
    # Divide every element in the array by the last value
    result_array_OM = [x / last_value_OM for x in state_6_OM]
    res_OM = np.array(result_array_OM)
    threshold = 1.0  # The value you want to find
    
    # Find the index when the threshold is crossed using NumPy
    indices_OM = np.where(res_OM >= threshold)
    
    if indices_OM[0].size > 0:
        first_crossing_index_OM = indices_OM[0][0]
        print(f"The threshold {threshold} is first crossed at index {first_crossing_index_OM}.")
    
    OM.append(first_crossing_index_OM)
    values_to_store_OM = res_OM[:first_crossing_index_OM].copy()
    
    
    
    x1 = np.arange(len(values_to_store_SM))
    x2 = np.arange(len(values_to_store_OM))
    
    # Plot both arrays on the same graph using different x-values
    plt.plot(x1, values_to_store_SM, label="SM")
    plt.plot(x2, values_to_store_OM, label="OM")
    
    end_of_shorter_array = len(values_to_store_SM) if len(values_to_store_SM) < len(values_to_store_OM) else len(values_to_store_OM)
    
    end_of_longer_array = len(values_to_store_OM) if len(values_to_store_SM) < len(values_to_store_OM) else len(values_to_store_SM)
    
    # Draw a vertical line at the end of the shorter array
    plt.axvline(x=end_of_shorter_array-1, color='r', linestyle='--', label=f'SM reaches convergence in {len(values_to_store_SM)-1} iteration ')
    
    plt.axvline(x=end_of_longer_array-1, color='b', linestyle='--', label=f'OM reaches convergence in {len(values_to_store_OM)-1} iteration ')
    
    # Add labels, legend, and title
    plt.xlabel("Iterations")
    plt.ylabel("Ratio with respect to converged value")
    plt.legend()
    plt.title(f"State {str(i)}")
    plt.savefig(f'plot {i}.png')
    # Show the plot
    plt.show()

data = {'State': state, 'OM_iteration': OM, 'SM_iteration': SM}
df_iteration = pd.DataFrame(data)
df_iteration.to_csv("8th Table.csv")















interval = 100
selected_rows = OM_State.iloc[::interval]

# Create a new DataFrame from the selected rows
new_OM_State = pd.DataFrame(selected_rows)

# Display the new DataFrame
new_OM_State.to_csv("New OM_3_1.csv")

interval = 100
selected_rows1 = SM_State.iloc[::interval]

# Create a new DataFrame from the selected rows
new_SM_State = pd.DataFrame(selected_rows1)

# Display the new DataFrame
new_SM_State.to_csv("New SM_3_1.csv")


plt.plot(OM_State["40"], label=f'DF1 {1}')
plt.plot(SM_State["40"], label=f'DF2 {1}')

plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()


#plt.plot(new_OM_State["16"], label=f'DF1 {4}')
#plt.plot(new_SM_State["16"], label=f'DF2 {4}')


series1=new_OM_State["16"]
series2 = new_SM_State["16"]
diff = series2-series1
o=len(diff)


fig, ax = plt.subplots()

# Plot the data
ax.plot(range(o), diff)

# Customize the spines (axes)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set the x-axis position to the middle of the y-axis range
ax.spines['bottom'].set_position(('data', 0))

# Customize other plot elements (labels, title, etc.) as needed
plt.xlabel('Number of occunrence of state 1')
plt.ylabel('Difference between SM and OM')


# Show the plot
plt.show()