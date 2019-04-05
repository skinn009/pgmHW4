from VertexColoringProblem import print_prob, sumprod, maxprod, EdgePotential, NodePotential
import numpy as np
import ast
import sys
from Graph import *
from BP import BP
from math import e
import string
import utilities
import random

def get_args (args):
    """
    Parses the string of arguments as presented on the commandline, and returns the adjacency matrix as a python
    and the samples matrix as an n x m array where n is the number of vertices and m is the number of samples,
    so that the [i,t] element represents the color of vertex i in sample t.
    """

    args = args[1:]
    args_list = ''.join(args).strip().replace(']][[',']]|[[').split('|')

    adjM = ast.literal_eval(args_list[0])
    samples = ast.literal_eval(args_list[1])
    return adjM, samples

class colormle_obj:

    def __init__(self, adj_matrix, examples):
        self.A = np.array(adj_matrix)
        self.examples = examples
        self.probs = list()
        self.w = []
        self.iterations = 100
        self.domain = self.get_domain()



    def get_domain(self):
        '''
        computes a list of possible values for x, by finding the largest value in the data.  We assume colors are 0 to
        the highest color-1
        :return:
        '''
        max_color = max([max(list) for list in self.examples])
        #print(max_color)
        return [i for i in range(max_color +1)][1:]

    def get_feature_vec(self, color):
        """
        Returns a vector of size of domain, with 1 indicating the color and 0's elsewhere.
        :param color:
        :return:
        """
        return utilities.list_to_vec([0 if color != item else 1 for item in self.domain])

    def get_prob_vec(self,vertex,p):
        prob_vec = [0 for i in range(len(self.domain))]
        for key in p[vertex].keys():
            prob_vec[key] = p[vertex][key][0]
        #print(prob_vec)
        #print()
        return utilities.list_to_vec(prob_vec)

    def compute_weights(self):
        if self.w == []:
            self.w = utilities.list_to_vec([1 for i in range(len(self.domain))])
            #print(self.domain)
            #print(self.w)
        eta = 1.0/len(self.examples[0])
        #print(self.domain)
        for i in range(self.iterations):

            #eta = eta/(i+1)
            #print("Iteration " + str(i))
            p = print_prob(self.A,self.w,10, max_prod=False)
            #print(p)
            grad = utilities.list_to_vec([0 for i in range(self.w.shape[0])])

            for vertex in range(self.A.shape[0]):
                p_vec = self.get_prob_vec(vertex,p)
                for sample_color in self.examples[vertex]:
                    s_vec = self.get_feature_vec(sample_color)
                    grad = grad + (s_vec - p_vec)

            self.w = self.w + eta*grad


        print ("w")
        print(self.w)

def colormle(A, samples):

    b = colormle_obj(A,samples)
    #print(b.examples)
    b.compute_weights()
    #print(b.probs)


''''
def make_graph (A,w,its = 50):
    A = np.array(A)
    n = len(A)

    domain = Domain(tuple(range(len(w))))
    edge_potential = EdgePotential()
    node_potential = NodePotential(w)

    rvs = list()
    factors = list()

    for i in range(n):
        rv = RV(domain, value=None)
        rvs.append(rv)
        factors.append(
            F(node_potential, (rv,))
        )

    for i in range(n):
        for j in range(n):
            if i < j and A[i, j] == 1:
                factors.append(
                    F(edge_potential, (rvs[i], rvs[j]))
                )

    g = Graph(rvs, factors)
    bp = BP(g, max_prod= False)
    bp.run(iteration = its)
    for f in bp.g.factors:

        #print(bp.factor_prob(f))

    #print("________________________")
    for rv in bp.g.rvs:
        #print(bp.prob(rv))
'''
if __name__ == "__main__":
    args = sys.argv
    mat, samples = get_args(args)
    A = np.array(mat)
    #w = [1,2]

    #ps = print_prob(A,w,10, max_prod=  False)


    colormle(mat,samples)

