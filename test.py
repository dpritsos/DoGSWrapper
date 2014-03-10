

import numpy as np

import sklearn.cross_validation as skcv

for i in skcv.StratifiedKFold(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]), 3, indices=True):
	print i