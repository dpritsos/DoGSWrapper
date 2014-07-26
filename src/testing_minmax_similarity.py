

import numpy as np

def minmax_similarity(v1, v2):

	return np.sum( np.min( np.vstack((v1,v2)), axis=0 ) ) / np.sum( np.max( np.vstack((v1,v2)), axis=0 ) )


c = np.array([1,1,1,0.01])
rv = np.array([0.1,0.9,0.1,0.1])

print np.min( np.vstack((c,rv)), axis=0 ) 
print minmax_similarity(c, rv)