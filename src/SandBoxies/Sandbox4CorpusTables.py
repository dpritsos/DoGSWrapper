	
import numpy as np #numpy.linalg for SVD etc.
#import numpy.linalg as alg
import scipy.spatial.distance as spd
import sklearn.decomposition as skd #sklearn.decomposition
import tables as tb
import time


corpus_mtrx_fname = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Kfolds_Vocs_Inds_4Chars_7Genres/kfold_CorpusMatrix_9100000.h5'

#Loading a table from any corpus available at a time. 
h5f = tb.open_file(corpus_mtrx_fname, 'r')
corpus_mtrx = h5f.get_node('/',  'corpus_earray')


#Testing Mahalanobis distance using randomly selected dimentions and timeing it. 
#Using projections of a random vector on new PC/Coordinates system.
def RandomMahal():

	

	print "COVARIANCE Manualy Made:"

	ts = time.time()

	#Calculate Variables Means
	mean_vect = np.mean(corpus_mtrx[0:100, 0:50000], axis=0)

	#Convert Raw data vectors to Zero-mean Vectors
	data_zero_mean = corpus_mtrx[0:100, 0:50000] - mean_vect
	

	#Display Manualy Crreated Covariance Matrix:
	print np.matrix(data_zero_mean).T.shape
	cov1 =  np.ascontiguousarray( np.matrix(data_zero_mean).T ) * np.matrix(data_zero_mean) / (np.matrix(data_zero_mean).shape[0] - 1)

	print cov1

	print np.linalg.eig(cov1)

	ts -= time.time()
	print ts
	
	print
	print "COVARIANCE"

	#ts = time.time()

	#print np.cov(corpus_mtrx[0:100, 0:50000].T)

	#ts -= time.time()
	#print ts

	

	print
	print "COVARIANCE form SVD"

	ts = time.time()

	U, S, V = np.linalg.svd(data_zero_mean, full_matrices = False)

	#print np.diag(S)

	print U.shape, np.matrix(np.diag(S)).shape, V.shape

	#print np.matrix(U).T.shape

	print len(S)

	#print np.ascontiguousarray( np.matrix(V).T ) * ( np.ascontiguousarray( np.matrix(np.diag(S)).T ) *  np.matrix(np.diag(S)) ) *  np.ascontiguousarray( np.matrix(V) ) / (np.matrix(data_zero_mean).shape[0] - 1)
	#print np.matrix(U).shape, (np.matrix(np.diag(S)) * np.matrix(np.diag(S)).T).shape, np.matrix(U).shape
	# np.matrix(U) * (np.matrix(np.diag(S)) * np.matrix(np.diag(S)).T) * np.matrix(U).T   #SAME AS#   print np.matrix(data_zero_mean) * np.matrix(data_zero_mean).T 

	ts -= time.time()
	print ts

	






RandomMahal()


#Testing Mahalanobis distnace using PCA/SVD for selecting the Principal Components. 
#Using projections of a random vector on new PC/Coordinates system.
def PCAMahal():
	
	pca = skd.PCA()
	print pca.fit(corpus_mtrx)

#PCAMahal()

#mean = [.1,.1,.1,.1,.1,.5,.1,.1,.1,.1,.1,.1,.6,.1,.1,.1,.1,.1,.1]
#cov = np.diag([.5,.7,.9,.2,.10,.5,.9,.8,.9,.1,.4,.1,.1,.6,.7,.3,.5,.8,.4])

#train_set = np.random.multivariate_normal(mean,cov,10)

#estim = np.mean(train_set, axis=0)

#print np.cov(train_set.T)
#print estim

#test_vetc = [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]

#pca = PCA()
#pca.fit(train_set)

#incov = alg.inv( alg.pinv(train_set.T) )



#print cov.shape
#print np.cov(train_set).shape
#print train_set.shape

#print incov

#test_vetc = pca.transform(test_vetc)
#mean = pca.transform(mean)


#print spd.mahalanobis(test_vetc,mean,incov)


#print incov
#print alg.pinv( np.cov(d) )
#print alg.svd(d)
#print l

h5f.close()




