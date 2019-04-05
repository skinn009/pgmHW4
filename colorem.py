from VertexColoringProblem import print_prob, sumprod, maxprod
import numpy as np
import ast
import sys
import utilities
import string
from sumProduct import sample_get_beliefs
import itertools

def get_args (args):
    """
    Parses the string of arguments as presented on the commandline, and returns the adjacency matrix as a python
    and the samples matrix as an n x m array where n is the number of vertices and m is the number of samples,
    so that the [i,t] element represents the color of vertex i in sample t.
    """

    args = args[1:]
    args_list = ''.join(args).strip().replace('][',']|[').split('|')

    adjM = ast.literal_eval(args_list[0])
    L = ast.literal_eval(args_list[1])
    samples = ast.literal_eval(args_list[2])
    return adjM, L, samples

class colorem_obj:

    def __init__(self, adj_matrix, L, examples):
        self.A = np.array(adj_matrix)
        self.examples = examples
        self.L = L
        self.probs = list()
        self.w = list()
        self.iterations = 50
        self.domain = self.get_domain()
        self.num_examples = len(self.examples[0])
        self.num_variables = self.A.shape[0]


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

    def get_tuple_product(self,tuple):
        prod = 1
        for item in tuple:
            prod = prod*item
        return prod

    def compute_phi_dict(self, sample):
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

    def wts_to_list(self):
        return [self.w[color][0] for color in range(self.w.shape[0])]

    def make_example(self,i):
        example = [None for i in range(self.num_variables)]
        for vertex in range(len(self.examples)):
            if self.L[vertex]==0:
                example[vertex] = self.examples[vertex][i]
        #print(example)
        return example

    def get_product_lists(self, p, beliefs, unobserved_vars):
        """
        Returns a list of the product permutations for the beliefs and probabilities.
        :param p:
        :param beliefs:
        :param unobserved_vars:
        :return:
        """
        p_arrays = []
        b_arrays = []
        for v in unobserved_vars[1:]:
            bel = beliefs[v]
            prob = self.get_prob_vec(v,p)
            p_arrays.append(prob)
            b_arrays.append(bel)
        bel_list = [[item[0] for item in b_array.tolist()] for b_array in b_arrays]
        prob_list = [[item[0] for item in p_array.tolist()] for p_array in p_arrays]
        bel_permutes = list(itertools.product(*bel_list))
        prob_permutes = list(itertools.product(*prob_list))

        p_products = [self.get_tuple_product(item) for item in prob_permutes]
        b_products = [self.get_tuple_product(item) for item in bel_permutes]

        return p_products, b_products

    def compute_weights(self):
        unobserved_list = [i for i, val in enumerate(self.L) if val ==1]
        if self.w == []:
            self.w = utilities.list_to_vec([1 for i in range(len(self.domain))])
        #print("unobs")
        #print(unobserved_list)
        eta = 1.0/self.num_examples

        for i in range(self.iterations):

            p = print_prob(self.A,self.w,10, max_prod=False)
            #print(p)
            grad = utilities.list_to_vec([0 for i in range(self.w.shape[0])])


            #First, we compute grad updates over observed variables
            for vertex in range(self.num_variables):

                if self.L[vertex] == 0:
                    p_vec = self.get_prob_vec(vertex,p)
                    for sample_color in self.examples[vertex]:
                        s_vec = self.get_feature_vec(sample_color)
                        grad = grad + (s_vec - p_vec)

                #The grad update for unobserved samples
                #for vertex, indicator in enumerate(self.L):
                if len(unobserved_list)==0:
                    break

                for vertex in unobserved_list:
                    p_vec = self.get_prob_vec(vertex,p)
                    #print("p_vec")
                    #print(p_vec)
                    for i in range(self.num_examples):
                        example = self.make_example(i)
                        phi_dict = self.compute_phi_dict(example)
                        beliefs = sample_get_beliefs(self.A,self.wts_to_list(),phi_dict,20)
                        #print(beliefs)
                        if len(unobserved_list) == 1:
                            s_vec = beliefs[vertex]
                            grad = grad + (s_vec - p_vec)

                        else:
                            s_vec = beliefs[vertex]
                            p_probs, belief_probs = self.get_product_lists( p, beliefs, unobserved_list)
                            for i in range(len(p_probs)):
                                grad = grad + (s_vec*belief_probs[i] - p_vec*p_probs[i])



            self.w = self.w + eta*grad
        #print(beliefs)
        print ("my w")
        print(self.w)


def colorem(A,L,samples):
    a = colorem_obj(A,L, samples)
    #print(a.L)
    a.compute_weights()





if __name__ == "__main__":
    args = sys.argv
    mat, L, samples = get_args(args)
    A = np.array(mat)
    #w = [1,2]
    args = sys.argv
    #ps = print_prob(A,w,10, max_prod=  False)


    colorem(mat,L,samples)
    print("helo")
