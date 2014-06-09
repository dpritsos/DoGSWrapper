

import numpy as np

from scipy.spatial.distance import cosine as cosine_dist

import timeit


cos_old = """\

def cosine_similarity(vector, centroid):

    #Convert Arrays and Matricies to Matrix Type for preventing unexpeted error, such as returning vector instead of scalar
    vector = np.matrix(vector)
    centroid = np.matrix(centroid)

    return vector * np.transpose(centroid) / ( np.linalg.norm(vector) * np.linalg.norm(centroid) )

"""

cos_new = """\

def cosine_sim(vector, centroid):

	#Convert Arrays and Matricies to Matrix Type for preventing unexpeted error, such as returning vector instead of scalar
	vector = np.matrix(vector)
	centroid = np.matrix(centroid)

	return 1 - cosine_dist(vector, centroid) 

"""

vector = np.array([0.5,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.1,0.6,0.8,0.7,0.8,0.20,0.0,0.0,0.0,0.0,0.0,0.0,0.0,])
centroid = np.array([0.15,0.31,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.21,0.11,0.61,0.18,0.17,0.81,0.20,0.0,0.0,0.0,0.0,0.01,0.0,0.0,])

#print (timeit.repeat(cos_old, number=100000) / timeit.repeat(cos_new, number=100000)) * 100
print ( np.mean( timeit.repeat(cos_new, repeat=1000) ) / np.mean( timeit.repeat(cos_old, repeat=1000) ) )

#print cosine_similarity(vector, centroid), cosine_sim(vector, centroid)