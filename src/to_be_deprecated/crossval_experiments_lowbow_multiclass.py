"""

"""
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')

import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb
#import html2vect.base.convert.tfttools as tbtls
#from html2vect.base.convert.tfdtools import TFDictTools
#from html2vect.base.convert.convert import TFVects2Matrix2D

import sklearn.decomposition as decomp
import sklearn.svm.libsvm_sparse as libsvm
import sklearn.svm as svm
import sklearn.svm.sparse as sp_svm
import sklearn.linear_model as linear_model
import sklearn.covariance as skcov
from sklearn.metrics import precision_score, recall_score
from sklearn import cross_validation

import scipy.sparse as ssp
from scipy import stats

from html2vect.sparse.lowbow import Html2LBN, Html2LBW, Html2LBN4SEG, Html2LBN_L1_BW
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.grid_search import GridSearchCV
  
class ResultsTable_desc(tb.IsDescription):
    kfold = tb.UInt32Col(pos=1)
    nu = tb.Float32Col(pos=2)
    feat_num = tb.UInt32Col(pos=3)
    F1 = tb.Float32Col(pos=4)
    P = tb.Float32Col(pos=5)
    R = tb.Float32Col(pos=6)
    TP = tb.Float32Col(pos=7)
    FP = tb.Float32Col(pos=8)
    TN = tb.Float32Col(pos=9)
    FN = tb.Float32Col(pos=10)
    
    
    
class MultiResultsTable_desc(tb.IsDescription):
    kfold = tb.UInt32Col(pos=1)
    Acc = tb.Float32Col(pos=2)
    feat_num_rq = tb.UInt32Col(pos=3)
    feat_num_rt = tb.UInt32Col(pos=4)
    Centroids_num = tb.UInt32Col(pos=5)
    Centroids = tb.Float32Col(shape=(1,10), pos=6)
    Sigma = tb.Float32Col(pos=7)
    Predicted_Y = tb.Float32Col(shape=(1,5000),pos=8)
    Expected_Y = tb.Float32Col(shape=(1,5000),pos=9)
    SVM_C = tb.Float32Col(pos=10)


class CSVM_CrossVal_Lowbow(object):
    
    def __init__(self, lowbow_type, h5f_res, corpus_path, genres):        
        self.lowbow = lowbow_type 
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.res_table = h5f_res.createTable(h5f_res.root, "multiclass_cross_validation",  MultiResultsTable_desc)
    
        
    def corpus_files_and_tags(self):
        xhtml_file_l = list()
        cls_gnr_tgs = list()
        for i, g in enumerate(genres):
            gnrs_file_lst = self.lowbow.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
            
            xhtml_file_l.extend( gnrs_file_lst )
                 
            cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )
                
        return (xhtml_file_l, cls_gnr_tgs)
    
    
    def evaluate(self, xhtml_file_l, cls_gnr_tgs, kfolds, C_lst, featr_size_lst, Kernel, Centroids_ll, Sigma_l):
        #Define SVM type
        
        #csvm = svm.LinearSVC(C=c)
        
        #Convert lists to Arrays
        xhtml_file_l = np.array( xhtml_file_l )
        cls_gnr_tgs = np.array( cls_gnr_tgs )
        
        #Starting CrossValidation
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, kfolds, indices=True)
        for k, (trn_idxs, crv_idxs) in enumerate(KF):
            
            print "Creating DICTIONARY "
            tf_d = dict() 
            #Merge All Term-Frequency Dictionaries created by the Raw Texts            
            for html_str in self.lowbow.load_files( list( xhtml_file_l[trn_idxs] ), encoding='utf8', error_handling='replace' ):
                tf_d = self.lowbow.merge_tfds(tf_d, self.lowbow.tf_dict( self.lowbow._attrib_(html_str) ) )
            
            #SELECT FEATURE SIZE 
            for featrs_size in featr_size_lst:
                resized_tf_d = self.lowbow.keep_atleast(tf_d, featrs_size) #<---
                print len(resized_tf_d)
                print resized_tf_d.items()[0:50]
            
                #Create The Terms-Index Dictionary that is shorted by Frequency descending order
                tid = self.lowbow.tf2tidx( resized_tf_d )
                print tid.items()[0:50]
                
                #SELECT CENTROIDS AND SIGMA
                for centroids_l in Centroids_ll:
                    for Sigma in Sigma_l:
                        print "Creating LOWBOW"
                        #Create LowBow Vectors Sparse Matrix
                        corpus_mtrx = self.lowbow.from_files( list( xhtml_file_l ),\
                                                              centroids_l, Sigma, tid_dictionary=tid,\
                                                              encoding='utf8', error_handling='replace' )
                        
                        #SELECT C
                        for c in C_lst:
                            csvm = sp_svm.SVC(C=1, kernel='linear', scale_C=False)
                            #csvm.set_params(C=c, kernel=Kernel, scale_C=False)
                            
                            print "FIT MODEL"
                            #FIT MODEL
                            train_Y = cls_gnr_tgs[ trn_idxs ] 
                            mtrx = corpus_mtrx[0]
                            train_X = mtrx[trn_idxs,:]   
                            #Some checks    
                            print ssp.issparse(train_X), train_X.shape[0], train_X.shape[1] #, len(train_Y), train_Y 
                            print ssp.isspmatrix_csr(train_X)
                            
                            csvm.fit( ssp.csr_matrix(train_X.todense(), shape=train_X.shape), train_Y) #<----- WATCH THIS
                            
                            print "EVALUATE"
                            #EVALUATE
                            crossval_Y = cls_gnr_tgs[ crv_idxs ]
                            mtrx = corpus_mtrx[0]
                            crossval_X = mtrx[crv_idxs,:]
                             
                            #ssp.csr_matrix(crossval_X, shape=crossval_X.shape, dtype=np.float64)
                            res_acc_score = csvm.score( ssp.csr_matrix(crossval_X.todense(), shape=crossval_X.shape), crossval_Y) #<----- WATCH THIS
                            
                            Predicted_Y = csvm.predict( ssp.csr_matrix(crossval_X.todense(), shape=crossval_X.shape) )
                            
                            print "Accuracy:", res_acc_score, " for:", k, c, len(resized_tf_d), centroids_l, Sigma, Predicted_Y 
                            
                            print "SAVE RESAULTS"
                             
                            self.res_table.row['kfold'] = k
                            self.res_table.row['Acc'] = res_acc_score
                            self.res_table.row['feat_num_rq'] = featrs_size
                            self.res_table.row['feat_num_rt'] = len(resized_tf_d)
                            self.res_table.row['Centroids_num'] = len(centroids_l)
                            z = np.zeros((1,10))
                            z[0,np.arange(len(centroids_l))] = np.array(centroids_l)
                            self.res_table.row['Centroids'] = z
                            self.res_table.row['Sigma'] = Sigma
                            z = np.zeros((1,5000))
                            z[0,np.arange(len(Predicted_Y))] = Predicted_Y
                            self.res_table.row['Predicted_Y'] = z
                            z = np.zeros((1,5000))
                            z[0,np.arange(len(crossval_Y))] = np.array(crossval_Y)
                            self.res_table.row['Expected_Y'] = z
                            self.res_table.row['SVM_C'] = c
                            self.res_table.row.append()
                            self.res_table.flush()

def binary(segments2matrix, trm_l_len):
    
    threshold=0.0
    
    #Sum up and return the sparse matrix for this string/text
    segments2matrix_sum = segments2matrix.sum(0)
        
    #Get Normalised Sum of Sums
    segments2matrix_sum_sums = segments2matrix_sum / np.max(segments2matrix_sum)    
    #Be Careful after the above division the Sparse Matrix is transformed to Matrix (non-sparse)
    
    #print "Normalising"
    #print segments2matrix_sum_sums

    #segments2matrix_sum_sums[ np.where( segments2matrix_sum_sums >= threshold) ] = 1.0
    segments2matrix_sum_sums[ np.where( segments2matrix_sum_sums == threshold) ] = 1/np.log(trm_l_len)    
    
    #print segments2matrix_sum_sums
    
    return ssp.csr_matrix( segments2matrix_sum_sums, shape=segments2matrix_sum_sums.shape, dtype=np.float32)


def deriv(segments2matrix, trm_l_len):
    
    #Sum up and return the sparse matrix for this string/text
    matrix_sum = segments2matrix.sum(0)
    
    deriv0 = matrix_sum[0, 1::] - matrix_sum[0, 0:-1] 
    
    #deriv1 = deriv0 - np.min(deriv0) 
        
    #Get Normalised Sum of Sums
    deriv = deriv0 / np.max(deriv0)    
    #Be Careful after the above division the Sparse Matrix is transformed to Matrix (non-sparse)
    
            
    #print segments2matrix_sum_sums
    return ssp.csr_matrix( deriv, shape=deriv.shape, dtype=np.float32)
                 

if __name__=='__main__':
        
    lowbow_N = Html2LBN(4, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm, norm_func=None)
    lowbow_N4SG = Html2LBN4SEG(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm, norm_func=None)
    lowbow_N4L2 = Html2LBN_L1_BW(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm, norm_func=None)
    lowbow_W = Html2LBW(attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm, norm_func=deriv) #binary
    
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_RES_LowBow&Derivative_Ngrams_FAST.h5', 'w')
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/CSVM_RES_LowBow_Ngrams_FAST.h5', 'w')
    
    csvm_crossval_lowbow = CSVM_CrossVal_Lowbow(lowbow_N ,crp_crssvl_res, corpus_filepath, genres)
    
    kfolds = 10
    C_lst = [1]
    featr_size_lst = [1000,3000,5000,10000,15000,20000,30000] #[100,300,500,700,1000,3000] #[100,200,300,400,500,600,700,800,900,1000,5000,10000,15000,20000,25000,30000] 
    #Kernel = 'rbf' #<----------------------SOS 
    Kernel = 'linear' #<----------------------SOS
    Centroids_ll = [ 
                     #[0.5],\
                     #[0.2, 0.3],\
                     #[0.2, 0.5, 0.8],\
                     #[0.2, 0.4, 0.6, 0.8],\
                     [0.1, 0.3, 0.5, 0.7, 0.9],\
                     #[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                   ] 
    Sigma_l = [0.2] #[0.1, 0.2, 0.5, 0.7, 0.9]
            
    xhtml_file_l, cls_gnr_tgs = csvm_crossval_lowbow.corpus_files_and_tags()
    
    csvm_crossval_lowbow.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, C_lst, featr_size_lst, Kernel, Centroids_ll, Sigma_l)
    
    crp_crssvl_res.close()

    
    
    
    
    
    
    
    
    
    
    

