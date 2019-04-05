'''
Generate samples given a probability distribution in the form of a list of dictionaries, where the ith element
of the list representing a distribution over the value i can take.

eg: [{0: 0.12708793631229953, 1: 0.2582238484002759, 2: 0.6146882152874247},
 {0: 0.25822384840027585, 1: 0.48355230319944825, 2: 0.25822384840027585},
  {0: 0.12708793631229953, 1: 0.2582238484002759, 2: 0.6146882152874247}]

The other parameter is the number of samples desired. The samples matrix is an n x m array where n is the number of vertices and m is the number of samples,
    so that the [i,t] element represents the color of vertex i in sample t.
'''
import random

class samples:

    def __init__(self, probs, n):
        self.probs = probs

        self.num_samples = n
        self.points = self.generate_samples()

    def generate_samples(self):
        #print(self.probs)
        point_list = [[] for i in range (len(self.probs[0]))]
        for i in range(self.num_samples):
            #sample = []
            for vert, probs in enumerate(self.probs):
                assignment = self.get_sample(probs)
                point_list[vert].append(assignment)
        return point_list



    def get_sample(self, probs):
        curr = 0
        rand = random.random()
        #print(rand)
        for key in probs.keys():
            curr = curr + probs[key]
            #print(curr)
            if rand < curr:
                assignment = key + 1
                break
        return assignment
