	
import numpy as np #numpy.linalg for SVD etc.
#import numpy.linalg as alg
import scipy.spatial.distance as spd
from sklearn.decomposition import PCA
import tables as tb


corpus_mtrx_fname = '/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Kfolds_Vocs_Inds_4Chars_7Genres/kfold_CorpusMatrix_9100000.h5'

#Loading a table from any corpus available at a time. 
h5f = tb.open_file(corpus_mtrx_fname, 'r')
corpus_mtrx = h5f.get_node('/',  'corpus_earray').read()


#Testing Mahalanobis distance using randomly selected dimentions and timeing it. 
#Using projections of a random vector on new PC/Coordinates system.
def RandomMahal():

	U, S, V = np.linalg.svd(corpus_mtrx[:,0:90000])

	#print U
	print S
	

RandomMahal()


#Testing Mahalanobis distnace using PCA/SVD for selecting the Principal Components. 
#Using projections of a random vector on new PC/Coordinates system.
def PCAMahal():
	pass

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




