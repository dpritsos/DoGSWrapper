	
import numpy as np
import numpy.linalg as alg
import scipy.spatial.distance as spd
from sklearn.decomposition import PCA


mean = [.1,.1,.1,.1,.1,.5,.1,.1,.1,.1,.1,.1,.6,.1,.1,.1,.1,.1,.1]
cov = np.diag([.5,.7,.9,.2,.10,.5,.9,.8,.9,.1,.4,.1,.1,.6,.7,.3,.5,.8,.4])

train_set = np.random.multivariate_normal(mean,cov,10)

estim = np.mean(train_set, axis=0)

print np.cov(train_set.T)

print estim

test_vetc = [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]

pca = PCA()
pca.fit(train_set)

incov = alg.inv( alg.pinv(train_set.T) )



#print cov.shape
#print np.cov(train_set).shape
#print train_set.shape

#print incov

#test_vetc = pca.transform(test_vetc)
#mean = pca.transform(mean)


print spd.mahalanobis(test_vetc,mean,incov)


#print incov
#print alg.pinv( np.cov(d) )
#print alg.svd(d)
#print l






