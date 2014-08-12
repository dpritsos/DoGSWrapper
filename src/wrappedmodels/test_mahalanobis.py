	
import numpy as np
import numpy.linalg as alg
import scipy.spatial.distance as spd
from sklearn.decomposition import PCA


mean = [.1,.1,.1,.3,.1,.1,.1,.8,.1,.1,.1,.1,.1,.1,.1,.1,.5,.8,.4]
cov = np.diag([.5,.7,.9,.2,.10,.5,.9,.8,.9,.1,.4,.1,.9,.6,.7,.3,.5,.8,.4])

train_set = np.random.multivariate_normal(mean,cov,10)

estim = np.mean(train_set, axis=0)

print estim

test_vetc = [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]

pca = PCA()
pca.fit(train_set)

incov = alg.inv( np.cov(train_set) )

print incov



print spd.mahalanobis(test_vetc,mean,incov)









#print incov
#print alg.pinv( np.cov(d) )
#print alg.svd(d)
#print l






