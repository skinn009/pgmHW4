from VertexColoringProblem import *
from colormle import *
from colorem import *
import numpy as np
import Samples
import MCMCDemo
from MCMC import *
from sumProduct import *
A = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]]
)

w = [1, 2, 3]

ps = print_prob(A, w, 10, max_prod=False)
#print(ps)
#print[np.array(ps[key]) for key in ps.keys()]
'''
for p in ps:
    print(p)


#sum_product(A,w,50)
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

    return(Graph(rvs, factors))
'''
samples = Samples.samples(ps, 200)

#print(samples.points)
#colormle(A, samples.points)


#colormle(A,samples.points)

l = [1,0,0]
colorem(A,l,samples.points)




