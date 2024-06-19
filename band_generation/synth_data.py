###
# Synthetic band data generator
###

# As we don't have access to DBs yet, set this up so that we have something to work from.

# Band: p(k) = a_0 + \sum_{j=2}^{21} a_j x^k for k \in (-\pi, \pi)
# Represented as vector in R^20

import csv 
import numpy as np

N_instances = 1000
L = 20
with open('fake_data.csv','w') as csvfile:
    data_writer = csv.writer(csvfile)

    for n in range(N_instances):
        row = np.random.normal(0,1,(L,))
        data_writer.writerow(row)