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
    C = tb.Float32Col(pos=2)
    feat_num = tb.UInt32Col(pos=3)
    Acc = tb.Float32Col(pos=7)

corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]

class CSVM_CrossVal(object):
    
    def __init__(self, h5f_res, corpus_path, genres):        
        self.lowbow_N = Html2LBN(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
        self.lowbow_N4SG = Html2LBN4SEG(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
        self.lowbow_N4L2 = Html2LBN_L1_BW(3, attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)
        self.lowbow_W = Html2LBW(attrib='text', lowercase=True, valid_html=False, smoothing_kernel=stats.norm)

        self.h5f_res = h5f_res
        self.corpus_path = corpus_path
        self.genres_lst = genres
        
        self.kfold_chnk = dict()
        self.page_lst_tb = dict()
        self.kfold_mod = dict()
        self.gnr2clss = dict()
        #self.tfv2matrix2d = TFVects2Matrix2D(DSize=3000)
                    
    def prepare_data(self):
        
        xhtml_file_l = list()
        for g in genres:
            xhtml_file_l.extend( self.lowbow_W.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] ) )
        
        self.cls_gnr_tgs = list()
        for i in range(len(genres)):
            self.cls_gnr_tgs.extend( [i+1]*200 )
            
        #Create Corpus Dictionary for the training Set
        train_set_files = xhtml_file_l[0:180] +\
                          xhtml_file_l[200:380] +\
                          xhtml_file_l[400:580] +\
                          xhtml_file_l[600:780] +\
                          xhtml_file_l[800:980] +\
                          xhtml_file_l[1000:1180] +\
                          xhtml_file_l[1200:1380]
                     
        tf_d = dict() 
        #Merge All Term-Frequency Dictionaries created by the Raw Texts                  
        for html_str in self.lowbow_W.load_files(train_set_files, encoding='utf8', error_handling='replace'):
            tf_d = self.lowbow_W.merge_tfds(tf_d, self.lowbow_W.tf_dict( self.lowbow_W._attrib_(html_str) ) )
            
        tf_d = self.lowbow_W.keep_atleast(tf_d, 15000) #<---
        print len(tf_d)
        print tf_d.items()[0:50]
        #Create The Terms-Index Dictionary that is shorted by Frequency descending order
        tid = self.lowbow_W.tf2tidx( tf_d )
        
        print tid.items()[0:50]
        #Create LowBow Vectors Sparse Matrix
        self.corpus_mtrx = self.lowbow_W.from_files( xhtml_file_l,\
                                                     [0.2, 0.5, 0.8], 0.2, tid_dictionary=tid,\
                                                     encoding='utf8', error_handling='replace' )

            
    def evaluate(self, kfolds, C_lst, featr_size_lst, tset_size):
        
        for featrs_size in featr_size_lst: 
            ##### FIND MORE OPTICAML USE IF POSIBLE
            #Keep the amount of feature required - it will keep_at_least as many as
            #feat_len = np.max(np.where(  kfold_Dictionary_TF_arr.read()['freq'] == kfold_Dictionary_TF_arr.read()['freq'][featrs_size] )[0])
            #the featrs_size keeping all the terms with same frequency the last term satisfies the featrs_size
            #print "Features Size:", feat_len      
            for c in C_lst:
                csvm = sp_svm.SVC(C=c, kernel='linear', scale_C=False)
                #csvm = svm.LinearSVC(C=c)
                #csvm = sp_svm.LinearSVC(C=c)
                #csvm = linear_model.SGDClassifier(n_iter=50, alpha=1e-5, n_jobs=1)
                print "FIT model"
                train_Y = self.cls_gnr_tgs[0:180] +\
                          self.cls_gnr_tgs[200:380] +\
                          self.cls_gnr_tgs[400:580] +\
                          self.cls_gnr_tgs[600:780] +\
                          self.cls_gnr_tgs[800:980] +\
                          self.cls_gnr_tgs[1000:1180] +\
                          self.cls_gnr_tgs[1200:1380]
                print train_Y
                train_X = ssp.vstack((self.corpus_mtrx[0][0:180,:],\
                                      self.corpus_mtrx[0][200:380,:],\
                                      self.corpus_mtrx[0][400:580,:],\
                                      self.corpus_mtrx[0][600:780,:],\
                                      self.corpus_mtrx[0][800:980,:],\
                                      self.corpus_mtrx[0][1000:1180,:],\
                                      self.corpus_mtrx[0][1200:1380,:]), format='csr')
                print ssp.issparse(train_X), train_X.shape[0], train_X.shape[1] #, len(train_Y), train_Y 
                print ssp.isspmatrix_csr(train_X)
                #print train_X
                csvm.fit( ssp.csr_matrix(train_X, shape=train_X.shape, dtype=np.float64), train_Y)
                #support, support_vectors, n_class_SV, sv_coef, intercept, label, probA, probB =\
                #libsvm.libsvm_sparse_train(X=ssp.csr_matrix(train_X, shape=train_X.shape, dtype=np.float64), svm_type=2, kernel='linear', nu=0.5)

                #print "Predict for kfold:k",k
                #crossval_X = crossval_earr_X[:, 0:feat_len]
                crossval_Y = self.cls_gnr_tgs[180:200] +\
                             self.cls_gnr_tgs[380:400] +\
                             self.cls_gnr_tgs[580:600] +\
                             self.cls_gnr_tgs[780:800] +\
                             self.cls_gnr_tgs[980:1000] +\
                             self.cls_gnr_tgs[1180:1200] +\
                             self.cls_gnr_tgs[1380:1400]
                print crossval_Y
                crossval_X = ssp.vstack((self.corpus_mtrx[0][180:200,:],\
                                         self.corpus_mtrx[0][380:400,:],\
                                         self.corpus_mtrx[0][580:600,:],\
                                         self.corpus_mtrx[0][780:800,:],\
                                         self.corpus_mtrx[0][980:1000,:],\
                                         self.corpus_mtrx[0][1180:1200,:],\
                                         self.corpus_mtrx[0][1380:1400,:]), format='csr') 
                
                #np.where( crossval_earr_X[:, 0:feat_len] > 0, crossval_earr_X[:, 0:feat_len], 0)
                #crossval_X[ np.nonzero(crossval_X) ] = 1  
            
                res_acc_score = csvm.score( ssp.csr_matrix(crossval_X, shape=crossval_X.shape, dtype=np.float64), crossval_Y)
                
                print "Accuracy:", res_acc_score 
                #res_table.row['kfold'] = k
                #res_table.row['C'] = c
                #res_table.row['feat_num'] = feat_len
                #res_table.row['Acc'] = res_acc_score
                #res_table.row.append()
        #res_table.flush()
         
    def exe(self):
        self.prepare_data()
        self.evaluate(1, [1], [1], 20)
    
    #def complementof_list(self, lst, excld_dwn_lim, excld_up_lim):
    #    if excld_dwn_lim == 0:
    #        return lst[excld_up_lim:]
    #    if excld_up_lim == len(lst):
    #        return lst[0:excld_dwn_lim]
    #    inv_lst = np.concatenate((lst[0:excld_dwn_lim], lst[excld_up_lim:]))
    #    return inv_lst
        

if __name__=='__main__':
    
    kfolds = 10
    nu_lst = [0.2, 0.8]
    featr_size_lst = [1000]
    crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/CSVM_LOWBOW_RES.h5', 'w')
            
    csvm_crossval = CSVM_CrossVal(crp_crssvl_res, corpus_filepath, genres)
    csvm_crossval.exe()
    
    crp_crssvl_res.close()

    
    
    
    
    
    
    
    
    
    
    

