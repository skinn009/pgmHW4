
import numpy as np
import utilities
import sys
import ast
import math


def compute_phi_dict( sample):
    """
    Creates a dictionary, whose keys are the vertices (0-k), and whose values are np arrays of 1s for hidden
    variables and a feature vec for the variables having a value from the sample.  The aim is to multiply
    the vector * w as the messages are being computed, to clamp the probability at the vertices having a value to 1
    for the variable taking that value.
    :param sample:
    :return:
    """

    phi_dict = {}
    for i, color in enumerate(sample):
        if color == None:
            phi_dict[i] = utilities.list_to_vec([1 for i in range(len(self.domain))])
        else:
            phi_dict[i] = self.get_feature_vec(color)
    return phi_dict


def make_np_array(array):
    return np.array(array)

def get_vertex_list(npAdjArray):
    """
    Creates a vertex labeling 1:length of the array depth, accepting an np array.
    :param npAdjArray:
    :return:
    """
    return [i  for i in range(npAdjArray.shape[0])]

def get_message_keys(npAdjArray):
    """
    Returns the keys for the messages from the adjacency array. The key is a concatenation of the target vertex and
    the source vertex.  For example, '12' is the key for the message from vertex 1 to vertex 2.
    :param npAdjArray:
    :return:
    """
    return [str(x)+str(y) for x, y in np.ndindex(npAdjArray.shape) if npAdjArray[x,y] == 1]

def get_inc_message_keys_by_vertex(messageKeyList, vertexList):
    """
    Creates a dictionary of vertex:[incoming message keys] for each vertex. For example, for 3 vertices that are
    completely connected in a loop, we return {1: ['21', '31'], 2: ['12', '32'], 3: ['13', '23']}.
    :param messageKeyList:
    :return:
    """
    vert_inc_message_keys = {i:[messageKeyList[j] for j in range(len(messageKeyList)) if messageKeyList[j][1]
                                 == str(i)] for i in vertexList}
    return vert_inc_message_keys

def initiaize_message_dict (messageKeys, k):
    """
    Initialize the messages to a vertical vector of 1s, in the length of the range of possible values, as an np array. For
    example, for a cycle of 3 completely connected vertices:
    {'12': array([[1],
       [1],
       [1]]), '13': array([[1],
       [1],
       [1]]), '21': array([[1],
       [1],
       [1]]), '23': array([[1],
       [1],
       [1]]), '31': array([[1],
       [1],
       [1]]), '32': array([[1],
       [1],
       [1]])}
       The key is the message name.
    :param messageKeys:
    :param k:
    :return:
    """
    message_dict = {key:utilities.list_to_vec([1 for i in range(len(k))]) for key in messageKeys}
    return message_dict

def normalize_messages(messageDict):
    """
    Normalize the messages so they each sum to 1.
    :param messageDict:
    :return:
    """
    sumElements = 0
    for key in messageDict:

        sumElements = 0
        sumElements += np.sum(messageDict[key])
        if sumElements != 0:
            messageDict[key] = messageDict[key]/sumElements
        else: messageDict[key] = messageDict[key]

    #for key in messageDict:
       # messageDict[key] = messageDict[key]/sumElements
    return messageDict

def update_messages(vertex_list,message_dict, w,phi_dict,psi):
    """
    First, we initialize psi to encode a matrix enforcing psi(i,j) = i if 1 != j, and 0 otherwise.
    Then, we multiply the incoming pointwise, which is an elementwise multiplication of the incoming
    message np arrays, and also multiplying in phi which is an np array of the weight vector, w. Finally,
    this vector is multiplied by the transpose of psi, to sum out over the incoming variables
    :param vertex_list:
    :param message_dict:
    :param w:
    :return:
    """
    """
    psi = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        for j in range(len(w)):
            if i != j:
                psi[i][j] = 1
            else:
                psi[i][j] = 0
    """
    for vertex in vertex_list:
        outgoing_message_keys = [key for key in message_dict.keys() if key[0] == str(vertex)]
        #print(outgoing_message_keys)
        incoming_message_keys = [key for key in message_dict.keys() if key[1] == str(vertex)]
        #print(phi_dict)
        for out_key in outgoing_message_keys:

            multiplied_messages = phi_dict[vertex]* w

            #print("phidice of vertex")
            #print(phi_dict[vertex])
            #multiplied_messages = utilities.list_to_vec(w)
            incoming_key_list = [in_key for in_key in incoming_message_keys if str(in_key)[0] != str(out_key)[1]]
            if len(incoming_key_list) > 0:
                for in_key in incoming_key_list:

                    multiplied_messages = multiplied_messages * message_dict[in_key]
            else:
                multiplied_messages = phi_dict[vertex]*w
    #print(psi)
    #print(multiplied_messages)
            m_out = psi.T@multiplied_messages

            message_dict[out_key] = m_out
        #print("After working with vertex "+ str(vertex))
    message_dict = normalize_messages(message_dict)
    return message_dict

def create_beliefs_dict (vertex_list, ordinality):
    """
    Create a beliefs dictionary, initializing belief for every vertex and every possible assignment to 1000. The key
    is the vertex number, and the value is an np array of depth (ordinality) and width 1.
    :param vertex_list:
    :return:
    """

    beliefs_dict = {}
    init_beliefs = np.ones((ordinality,1))
    for i in range (ordinality):
        init_beliefs[i][0] = 1000
    for vertex in vertex_list:
        beliefs_dict[vertex] = init_beliefs
    return beliefs_dict

def compute_edge_beliefs(message_dict, w, psi, phi_dict):
    """
    psi = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        for j in range(len(w)):
            if i != j:
                psi[i][j] = 1
            else:
                psi[i][j] = 0
    """
    edge_belief_dict = {}
    w_f = [float(item) for item in w]
    phi = np.exp(utilities.list_to_vec(w_f))

    for key in message_dict.keys():
        i,j = key[0], key[1]
        mult_i = phi
        mult_j = phi

        incoming_keys_to_i = [key_in for key_in in message_dict.keys() if key_in[1] == i and key_in[0] != j]
        incoming_keys_to_j = [key_in for key_in in message_dict.keys() if key_in[1] == j and key_in[0] != i]

        diag_i = np.zeros((len(w),len(w)))
        if len(incoming_keys_to_i)>0:
            for key_in in incoming_keys_to_i:
                mult_i = mult_i * message_dict[key_in]
                diag_i = np.diagflat(mult_i)
        else: diag_i = np.diagflat(mult_i)
        if len(incoming_keys_to_j)>0:
            for key in incoming_keys_to_j:
                mult_j = mult_j * message_dict[key]
                diag_j = np.diagflat(mult_j)
        else: diag_j = np.diagflat(mult_j)
        belief_matrix = diag_i@(psi@diag_j)
        edge_belief_dict[key] = belief_matrix/np.sum(belief_matrix)

    return edge_belief_dict


def compute_new_beliefs (vertex_list, message_dict, w,phi_dict):
    """
    New beliefs are computed for each vertex as a product of phi[vertex] ,w, and the incoming messages
    :param vertex_list:
    :param message_dict:
    :param phi:
    :return:
    """
    new_belief_dict = {}
    #print("phidict in compute new beliefs")
    #print(phi_dict)
    wts = utilities.list_to_vec([float(item) for item in w])
    #print("wts in compute new beliefs")
    #print(wts)
    for vertex in vertex_list:
        incoming_message_keys = [key for key in message_dict if key[1] == str(vertex)]
        mult = wts*phi_dict[vertex]
        if len(incoming_message_keys) >0:
            for key in incoming_message_keys:
                mult = mult*message_dict[key]


        if np.sum(mult) != 0:
            belief = mult/np.sum(mult)


        else: belief = mult
        new_belief_dict[vertex] = belief
    #print("beleif dict")
    #print(new_belief_dict)
    return new_belief_dict

def beliefs_converge(cur_beliefs, prev_beliefs):
    """
    We compute the sum of the norms of the differences in the messages. Return true if < epsilon.
    :param cur_beliefs:
    :param prev_beliefs:
    :return:
    """
    epsilon = 0.0001
    sum_norms = 0
    for key in cur_beliefs:
        diff = np.linalg.norm(cur_beliefs[key]-prev_beliefs[key])
        sum_norms+=diff
    return sum_norms < epsilon

def compute_bethe_energy(vertex_beliefs, edge_beliefs, psi,phi):
    """
    The parameters are dictionaries: vertex:belief vector, edge:belief matrix. As written this only works for
    pairwise MRFs. To generalize, the edge term computation must be modified to allow more than one neighbor to
    the vertices.
    :param vertex_beliefs:
    :param edge_beliefs:
    :return:
    """
    vertex_term = 0
    edge_term = 0
    clique_term = 0
    #Compute the first term, over the vertex beliefs = 0
    #print(edge_beliefs.keys())
    for key in edge_beliefs.keys():

        for i in range (psi.shape[0]):
            for j in range (psi.shape[1]):
                clique_term += math.log(psi[i][j]**edge_beliefs[key][i][j])
    #print("clique edge term")
    #print(clique_term)
    """
    #new vertex term
    for key in vertex_beliefs.keys():
        for i in range(vertex_beliefs[key].shape[0]):
            vertex_term += vertex_beliefs[key][i]* (math.log(phi[i] - math.log(vertex_beliefs[key][i])))
    """
    #old clique term for vertices where the problem lies for loopy graphs
    for key in vertex_beliefs.keys():
        for i in range (vertex_beliefs[key].shape[0]):
            #print(vertex_beliefs[key][i])
            #print (np.log(phi[i][0]))
            clique_term += vertex_beliefs[key][i] * np.log(phi[i][0])
            #print(clique_term)
    
    #print("clique term")
    #print(clique_term)
    #print(vertex_beliefs)
    #Old vertex term
    for key in vertex_beliefs.keys():
        vertex_term+=(vertex_beliefs[key].T@np.log(vertex_beliefs[key]))[0][0]
        #print(vertex_term)

    for key in edge_beliefs:
        #print(edge_beliefs)
        vert_i,vert_j = int(float(key[0])),int(float(key[1]))
        if vert_i > vert_j:
            for i in range(edge_beliefs[key].shape[0]):
                for j in range(edge_beliefs[key].shape[1]):
                    belief = edge_beliefs[key][i][j]

                    if belief == 0:
                        edge_term += 0
                    else:
                        edge_term += (belief*np.log(belief) - belief*np.log(vertex_beliefs[vert_i][i])
                                 - belief*np.log(vertex_beliefs[vert_j][j]))

    #print(edge_term)
    """
    ##from the paper:
    new_edge_term = 0
    new_vertex_term = 0
    for key in vertex_beliefs.keys():
        q = len([edgeKey for edgeKey in edge_beliefs.keys() if edgeKey.endswith(str(key))]) -1
        new_vertex_term = new_vertex_term + q*vertex_beliefs[key].T@(np.log(vertex_beliefs[key]) - np.log(phi))
    for key in edge_beliefs.keys():
        vert_i, vert_j = int(key[0]), int(key[1])
        for i in range(edge_beliefs[key].shape[0]):
                for j in range(edge_beliefs[key].shape[1]):
                    e_belief = edge_beliefs[key][i][j]
                    new_edge_term = new_edge_term + math.log(e_belief**e_belief) - math.log(psi[i][j]**e_belief)\
                                    - math.log(vertex_beliefs[vert_i][0]**e_belief) - math.log(vertex_beliefs[vert_j][0]**e_belief)
    print (new_edge_term)

    print("new edge")
    print(new_vertex_term[0][0])
    print(math.exp(new_edge_term - new_vertex_term[0][0]))
        #print("clique term")
        #print(clique_term)
    """
    return -(vertex_term + edge_term) + clique_term
    #return vertex_term - edge_term + clique_term


def sum_product(adj_array, w, iter_num):
    """
    NEEDS TO BE REFACTORED LIKE 'SAMPLE_GET_BELIEFS' TO INCLUDE A PHI-DICT, WHICH WILL FOR EACH VARIABLE
    BE A VECTOR OF 1'S, DENOTING THAT NO VARIABLES HAVE ASSIGNED VALUES.
    :param adj_array:
    :param w:
    :param iter_num:
    :return:
    """
    adjArray = np.array(adj_array)
    vert_list = get_vertex_list(adjArray)
    message_keys = get_message_keys(adjArray)
    message_dict = initiaize_message_dict (message_keys, w)
    message_dict = normalize_messages(message_dict)
    psi = np.zeros((len(w), len(w)))
    bel_old = create_beliefs_dict(vert_list,len(w))
    phi = np.exp(utilities.list_to_vec(w))
    #print("shape phi")
    #print(phi.shape)
    #print(phi[0])
    for i in range(len(w)):
        for j in range(len(w)):
            if i != j:
                psi[i][j] = 1
            else:
                psi[i][j] = 0

    for i in range(iter_num):
        message_dict = update_messages(vert_list,message_dict,phi,psi)

        vertex_beliefs = compute_new_beliefs(vert_list,message_dict,w)
        #print("vert belief")
        #print(vertex_beliefs)
        print("Iteration number " + str(i))
        converge = beliefs_converge(vertex_beliefs, bel_old)
        print ("Belief convergence: "+ str(converge))
        bel_old = vertex_beliefs
        #print(bel)
    #print(vertex_beliefs)
    edge_beliefs = compute_edge_beliefs(message_dict,w, psi)
    #print("edge bel")
    #print(edge_beliefs)
    log_Z = compute_bethe_energy(vertex_beliefs,edge_beliefs, psi,phi)
    print()
    print("And the partition coefficent, Z is: " + str(math.exp(log_Z)))

def sample_get_beliefs(adj_array, w, phi_dict, iter_num=20):
    """
    adjArray = [[0,1,0],[1,0,1],[0,1,0]]
    wts = [1,1]
    Tests for convergence on a simple tree: v1--v2--v3
    :param adj_array:
    :param wts:
    :param iter_num:
    :return:
    """
    adjArray = np.array(adj_array)
    vert_list = get_vertex_list(adjArray)
    #print("vertex list")
   # print(vert_list)
    message_keys = get_message_keys(adjArray)
    message_dict = initiaize_message_dict (message_keys, w)
    message_dict = normalize_messages(message_dict)
    #print("Initial message dict")
    #print(message_dict)
    psi = np.zeros((len(w), len(w)))
    #print("psi in test sum product")
    for i in range(len(w)):
        for j in range(len(w)):
            if i != j:
                psi[i][j] = 1
            else:
                psi[i][j] = 0


    bel_old = create_beliefs_dict(vert_list,len(w))
    #print("initial belief dict")
    #print(bel_old)
    #psi = [[1,.2],[.2, .5]]
    psi = np.array(psi)
    wts = utilities.list_to_vec(w)

    for i in range(iter_num):
        message_dict = update_messages(vert_list,message_dict,wts,phi_dict,psi)
        vertex_beliefs = compute_new_beliefs(vert_list,message_dict,w,phi_dict)
        #print("Iteration number " + str(i))
        converge = beliefs_converge(vertex_beliefs, bel_old)
        #print ("Belief convergence: "+ str(converge))
        bel_old = vertex_beliefs
        #print("vertex beliefs")
#print(vertex_beliefs)
    return vertex_beliefs

    """
    edge_beliefs = compute_edge_beliefs(message_dict,w, psi)
    log_Z = compute_bethe_energy(vertex_beliefs,edge_beliefs, psi)
    print()
    print("And the partition coefficent, Z is: " + str(math.exp(log_Z)))
    """
#def compute_partition_function(belief_dict):
if __name__ == '__main__':
    #parse arguments:
    """
    TO start from commmand line: python3 sumProduct.py [[0,1,1],[1,0,1],[1,1,0]] [1,2,3] 20
    """
    args = sys.argv
    array = args[1]
    wts = args[2]
    iterations = int(float(args[3]))
    adjM = ast.literal_eval(array)
    w = ast.literal_eval(wts)
    values = 1

#sum_product(adjM, w, iterations)
    phi_dict = {0:utilities.list_to_vec([1,1,1]),1:utilities.list_to_vec([1,0,0]),2:utilities.list_to_vec([0,0,1])}
    a = sample_get_beliefs(adjM, w,phi_dict, iterations)
    print(a)


