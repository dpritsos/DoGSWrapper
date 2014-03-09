"""

"""

import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb

import scipy.sparse as ssp
import scipy.spatial.distance as spd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation
import sklearn.svm as svm
#import sklearn.svm.sparse as sp_svm

import html2vect.sparse.wngrams as h2v_wcng
import html2vect.sparse.cngrams as h2v_cng


        
class CrossVal_OCSVM(object):
    
    def __init__(self, TF_TT, h5_res, corpus_path, genres):
        self.TF_TT = TF_TT
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        self.h5_res = h5_res
        
                
    def corpus_files_and_tags(self):
        xhtml_file_l = list()
        cls_gnr_tgs = list()
        for i, g in enumerate(self.genres_lst):
            gnrs_file_lst = self.TF_TT.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
            
            xhtml_file_l.extend( gnrs_file_lst )
                 
            cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )
                
        return (xhtml_file_l, cls_gnr_tgs)
    
                      
    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, nu):
        inds_per_gnr = dict()
        inds = list()
        last_gnr_tag = 1
        
        for trn_idx in trn_idxs:
            
            if cls_gnr_tgs[trn_idx] != last_gnr_tag:
                inds_per_gnr[ self.genres_lst[last_gnr_tag - 1] ] = inds
                last_gnr_tag = cls_gnr_tgs[trn_idx]
                inds = []
            
            inds.append( trn_idx )    
        
        inds_per_gnr[ self.genres_lst[last_gnr_tag - 1] ] = inds 
    
        gnr_classes = dict()
        for g in self.genres_lst:
            
            #Create the OC-SVM Model for this genre
            gnr_classes[g] = svm.OneClassSVM(kernel='linear', nu=nu, shrinking=True, cache_size=200, verbose=False)
            print "Fit Model for ", g
            print "Corpus_Mtrx", corpus_mtrx[inds_per_gnr[g], :].shape
            #Convert TF vectors to Binary 
            #crp_arr_bin = np.where(corpus_mtrx[inds_per_gnr[g], :].toarray() > 0, 1, 0)
            
            #Fit OC-SVM Model to Data of this genre
            gnr_classes[g].fit( corpus_mtrx[inds_per_gnr[g], :].toarray() )
        
        return (gnr_classes, inds_per_gnr)   
    
    
    def predict(self, gnr_classes, crossval_X):
            
        #Initialise Predicted-Classes-Arrays List 
        predicted_Y_per_gnr = list()
        predicted_Dist_per_gnr = list()
        
        for g in self.genres_lst:
            
            #Convert TF vectors to Binary
            #cv_arr_bin = np.where(crossval_X.toarray() > 0, 1, 0)
            
            #Get the predictions for each Vector for this genre
            predicted_Y = gnr_classes[ g ].predict( crossval_X.toarray() ) #For an one-class model, +1 or -1 is returned.
            predicted_D = gnr_classes[ g ].decision_function( crossval_X.toarray() )
            
            #Keep the prediction per genre 
            predicted_Y_per_gnr.append( predicted_Y )
            predicted_Dist_per_gnr.append( predicted_D.reshape( predicted_D.shape[0] ) ) 
            
        #Convert it to Array before returning
        predicted_Y_per_gnr = np.vstack( predicted_Y_per_gnr )
        predicted_Dist_per_gnr = np.vstack( predicted_Dist_per_gnr )
    
        return (predicted_Y_per_gnr, predicted_Dist_per_gnr)      
        
    
    def evaluate(self, xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, nu_l, featr_size_lst, norm_func):
        
        #Convert lists to Arrays
        xhtml_file_l = np.array( xhtml_file_l )
        cls_gnr_tgs = np.array( cls_gnr_tgs )
        
        #Starting CrossValidation
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, kfolds, indices=True)
        for k, (trn_idxs, crv_idxs) in enumerate(KF):
            
            #Creating a Group for this k-fold in h5 file
            kfld_group = self.h5_res.createGroup('/', 'KFold'+str(k), "K-Fold group of Results Arrays" )
            
            print "Creating VOCABULARY" 
            #Creating Dictionary      
            tf_d = self.TF_TT.build_vocabulary( list( xhtml_file_l[trn_idxs] ), encoding='utf8', error_handling='replace' )      
            print list(tf_d)[0]
                 
            #SELECT VOCABILARY SIZE 
            for vocab_size in vocabilary_size:
                resized_tf_d = self.TF_TT.tfdtools.keep_atleast(tf_d, vocab_size) #<---
                print len(resized_tf_d)
                print resized_tf_d.items()[0:50]
                
                #Create The Terms-Index Vocabulary that is shorted by Frequency descending order
                tid = self.TF_TT.tfdtools.tf2tidx( resized_tf_d )
                print tid.items()[0:50]
                
                print "Creating Sparse TF Matrix for CrossValidation"
                #Create Sparse TF Vectors Sparse Matrix
                corpus_mtrx = self.TF_TT.from_files( list( xhtml_file_l ), tid_dictionary=tid, norm_func=norm_func,\
                                                         encoding='utf8', error_handling='replace' )
                
                #SELECT FREATUR SIZE
                for featrs_size in featr_size_lst:
                    
                    #Creating a Group for this features size in h5 file under this k-fold
                    feat_num_group = self.h5_res.createGroup(kfld_group, 'Feat'+str(featrs_size), "Features Number group of Results Arrays for this K-fold" )
                    
                    #SELECT DIFERENT nu Parameter
                    for nu in nu_l:
                        
                        print "Construct classes"
                        #Construct Genres Class Vectors form Training Set - With the proper featrs_size
                        gnr_classes, inds_per_gnr = self.contruct_classes(trn_idxs, corpus_mtrx[0][:,0:featrs_size], cls_gnr_tgs, nu)
                        
                        #SELECT Cross Validation Set - With the proper featrs_size
                        crossval_Y = cls_gnr_tgs[ crv_idxs ]
                        mtrx = corpus_mtrx[0]
                        crossval_X = mtrx[crv_idxs, 0:featrs_size] 
                                
                        print "EVALUATE"
                        #Creating a Group for this number of iterations in h5 file under this features number under this k-fold
                        nu_group = self.h5_res.createGroup(feat_num_group, 'Nu'+str(nu), "OC-SVM's Nu parameter group of Results Arrays for this K-fold" )
                       
                        predicted_Y_per_gnr, predicted_Dist_per_gnr = self.predict(gnr_classes, crossval_X) 
                        
                        print np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))
                        
                        
                        cv_tg_idxs = np.array( np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
                        
                        
                        P_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        R_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        F1_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        
                        end = 0
                        for gnr_cnt in range(len(self.genres_lst)):
                            start = end
                            end = end + cv_tg_idxs[gnr_cnt+1]
                            
                            #Counts per Genre (True Positive)
                            tp_cnts_per_gnr = np.sum( np.where( predicted_Y_per_gnr[gnr_cnt, start:end] > 0, 1, 0) ).astype(np.float)
                            #Count True Positive and False Positive per Genre
                            tp_n_fp = np.sum( np.where( predicted_Y_per_gnr[gnr_cnt, :] > 0, 1, 0) ).astype(np.float) 
                            #Count True Positive and False Negative Per Genre
                            #Acctualy this is equal to the number of samles is keept in the CrossValidation set per Genre
                            #i.e. cv_tg_idxs
                        
                            #Calculate Precision
                            P_per_gnr[gnr_cnt+1] = tp_cnts_per_gnr / tp_n_fp
                            print "P", P_per_gnr
                            #Calculate Recall
                            R_per_gnr[gnr_cnt+1] = tp_cnts_per_gnr / cv_tg_idxs[gnr_cnt+1]
                            print "R", R_per_gnr
                            #Calculate F1 score 
                            F1_per_gnr[gnr_cnt+1] = 2 * P_per_gnr[gnr_cnt+1] * R_per_gnr[gnr_cnt+1] / (P_per_gnr[gnr_cnt+1] + R_per_gnr[gnr_cnt+1])
                            print "F1", F1_per_gnr 
                        
                        #Zero Position cannot be used for OC-SVM as in Koppel's method because the later returns a unique class for each\
                        #page while OC-SVM can return more than one 
                        P_per_gnr[0] = 0  #precision_score(crossval_Y, predicted_Y)   
                        R_per_gnr[0] = 0 #recall_score(crossval_Y, predicted_Y) 
                        F1_per_gnr[0] = 0 #f1_score(crossval_Y, predicted_Y)  
                        
                        #Maybe Later
                        #fpr, tpr, thresholds = roc_curve(crossval_Y, predicted_Y)   
            
                        print self.h5_res.createArray(nu_group, 'expected_Y', crossval_Y, "Expected Classes per Document (CrossValidation Set)")[:]
                        print "Predicted_Y_per_gnr"                                         
                        print self.h5_res.createArray(nu_group, 'predicted_Y_per_gnr', predicted_Y_per_gnr, "Predicted Y OC-SVM results per Document per nu (CrossValidation Set)")[:]
                        print "predicted_Dist_per_gnr"
                        print self.h5_res.createArray(nu_group, 'predicted_Dist_per_gnr', predicted_Dist_per_gnr, "predicted Distance of the samples X to the separating hyperplane for OC-SVM results per Document per nu (CrossValidation Set)")[:]              
                        print self.h5_res.createArray(nu_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0] == 0)")[:]
                        print self.h5_res.createArray(nu_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0] == 0)")[:]
                        print self.h5_res.createArray(nu_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0] == 0)")[:]
                        print                
                                        
                                        

if __name__ == '__main__':
    
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'w')
    #CrossVal_OCSVM_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-OC-SVM_kfolds-10_TM-TF_(DIST).h5', 'w')
    CrossVal_OCSVM_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Char4Grams-OC-SVM_kfolds-10_TM-TF_(DIST)_TESTESTEST.h5', 'w')
    
    kfolds = 10
    vocabilary_size = [100000] #[1000,3000,10000,100000]
    nu_l = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
    featr_size_lst = [1000, 5000, 10000, 70000] #[1000, 5000, 10000, 20000, 50000, 70000]
    N_Gram_size = 4
    W_N_Gram_size = 1
    
     #sparse_WNG = h2v_wcng.Html2TF(W_N_Gram_size, attrib='text', lowercase=True, valid_html=False)
    sparse_CNG = h2v_cng.Html2TF(N_Gram_size, attrib='text', lowercase=True, valid_html=False)
    
    crossV_OCSVM = CrossVal_OCSVM(sparse_CNG, CrossVal_OCSVM_res, corpus_filepath, genres)
    
    xhtml_file_l, cls_gnr_tgs = crossV_OCSVM.corpus_files_and_tags()
    
    crossV_OCSVM.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, nu_l, featr_size_lst, norm_func=None)
    
    CrossVal_OCSVM_res.close()
