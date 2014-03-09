"""

"""

import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../../html2vectors/src')
import numpy as np
import tables as tb
#import html2tf.tables.cngrams as cng_tb

import scipy.sparse as ssp
import scipy.spatial.distance as spd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation

#import html2vect.sparse.wngrams as h2v_wcng
import html2vect.sparse.cngrams as h2v_cng


class CrossVal_Koppels_method(object):
    
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
    
                      
    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs):
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
            #Merge All Term-Frequency Dictionaries created by the Raw Texts
            gnr_classes[g] = corpus_mtrx[inds_per_gnr[g], :].mean(axis=0) #<-------
        
        return (gnr_classes, inds_per_gnr)   
    
    
    def predict(self, gnr_classes, crossval_X, crossval_Y, vocab_index_dct, featrs_size, similarity_func, sim_min_value, iters, sigma_threshold):
            
        max_sim_scores_per_iter = np.zeros((iters, crossval_X.shape[0]))
        predicted_classes_per_iter = np.zeros((iters, crossval_X.shape[0]))
                    
        print "FEATURE SIZE: ",featrs_size

        #Measure similarity for iters iterations i.e. for iters different feature subspaces Randomly selected 
        for I in range(iters):
            
            #Randomly select some of the available features
            suffled_vocabilary_idxs = np.random.permutation( np.array(vocab_index_dct.values()) ) 
            features_subspace = suffled_vocabilary_idxs[0:featrs_size]
            
            #Initialised Predicted Classes and Maximum Similarity Scores Array for this i iteration 
            predicted_classes = np.zeros( crossval_X.shape[0] )
            max_sim_scores = np.zeros( crossval_X.shape[0] )
            
            #Measure similarity for each Cross-Validation-Set vector to each available Genre Class(i.e. Class-Vector). For This feature_subspace
            for i_vect, vect in enumerate(crossval_X[:, features_subspace]):
                
                #Convert TF vectors to Binary 
                #vect_bin = np.where(vect[:, :].toarray() > 0, 1, 0) #NOTE: with np.where Always use A[:] > x instead of A > x in case of Sparse Matrices
                #print vect.shape
                
                max_sim = sim_min_value
                for cls_tag, g in enumerate(self.genres_lst):
                    
                    #Convert TF vectors to Binary
                    #gnr_cls_bin = np.where(gnr_classes[ g ][:, features_subspace] > 0, 1, 0)
                    #print gnr_cls_bin.shape
                    
                    #Measure Similarity
                    sim_score = similarity_func(vect, gnr_classes[ g ][:, features_subspace])
                    
                    #Just for debugging for 
                    #if sim_score < 0.0:
                    #    print "ERROR: Similarity score unexpected value ", sim_score
                    
                    #Assign the class tag this vector is most similar and keep the respective similarity score
                    if sim_score > max_sim:
                        predicted_classes[i_vect] = cls_tag + 1 #plus 1 is the real class tag 0 means uncategorised
                        max_sim_scores[i_vect] = sim_score
                        max_sim = sim_score
        
            #Store Predicted Classes and Scores for this i iteration
            max_sim_scores_per_iter[I,:] = max_sim_scores[:]
            predicted_classes_per_iter[I,:] = predicted_classes[:]
                                              
        predicted_Y = np.zeros((crossval_Y.shape[0]), dtype=np.float)
        predicted_scores = np.zeros((crossval_Y.shape[0]), dtype=np.float)
        
        for i_prd_cls, prd_cls in enumerate(predicted_classes_per_iter.transpose()):
            genres_occs = np.histogram( prd_cls.astype(np.int), bins=np.arange(self.gnrs_num+2))[0] #One Bin per Genre plus one i.e the first to be always zero
            #print genres_occs
            genres_probs = genres_occs.astype(np.float) / np.float(iters)
            #print genres_probs
            if np.max(genres_probs) >= sigma_threshold:
                predicted_Y[i_prd_cls] = np.argmax( genres_probs )
                predicted_scores[i_prd_cls] = np.max( genres_probs ) 
        
        return predicted_Y, predicted_scores, max_sim_scores_per_iter, predicted_classes_per_iter      
        
    
    def evaluate(self, xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst, sigma_threshold, similarity_func, sim_min_val, norm_func):
        
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

            #tf_d = dict() 
            #Merge All Term-Frequency Dictionaries created by the Raw Texts            
            #for html_str in self.TF_TT.load_files( list( xhtml_file_l[trn_idxs] ), encoding='utf8', error_handling='replace' ):
            #    tf_d = self.TF_TT.tfdtools.merge_tfds(tf_d, self.TF_TT.s2tf.tf_dict( self.TF_TT._attrib(html_str) ) )
                 
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

                print "Construct classes"
                #Construct Genres Class Vectors form Training Set
                gnr_classes, inds_per_gnr = self.contruct_classes(trn_idxs, corpus_mtrx[0], cls_gnr_tgs)
                
                #SELECT Cross Validation Set
                crossval_Y = cls_gnr_tgs[ crv_idxs ]
                mtrx = corpus_mtrx[0]
                crossval_X = mtrx[crv_idxs, :] 
                                
                #SELECT FREATUR SIZE
                for featrs_size in featr_size_lst:
                    
                    #Creating a Group for this features size in h5 file under this k-fold
                    feat_num_group = self.h5_res.createGroup(kfld_group, 'Feat'+str(featrs_size), "Features Number group of Results Arrays for this K-fold" )
                    
                    #SELECT DIFERENT ITERATIONS NUMBER
                    for iters in iter_l:
                        print "EVALUATE"
                        
                        #Creating a Group for this number of iterations in h5 file under this features number under this k-fold
                        iters_group = self.h5_res.createGroup(feat_num_group, 'Iters'+str(iters), "Number of Iterations (for statistical prediction) group of Results Arrays for this K-fold" )
                       
                        predicted_Y,\
                        predicted_scores,\
                        max_sim_scores_per_iter,\
                        predicted_classes_per_iter = self.predict(gnr_classes,\
                                                                  crossval_X, crossval_Y,\
                                                                  tid, featrs_size,\
                                                                  similarity_func, sim_min_val,\
                                                                  iters, sigma_threshold) 
                        
                        print np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))
                        print np.histogram(predicted_Y.astype(np.int), bins=np.arange(self.gnrs_num+2))
                        
                        cv_tg_idxs = np.array( np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
                        tp_n_fp = np.array( np.histogram(predicted_Y.astype(np.int), bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
                        
                        P_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        R_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        F1_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                        
                        end = 0
                        for gnr_cnt in range(len(self.genres_lst)):
                            start = end
                            end = end + cv_tg_idxs[gnr_cnt+1]
                            counts_per_grn_cv = np.histogram( predicted_Y[start:end], bins=np.arange(self.gnrs_num+2) )[0]
                            #print counts_per_grn_cv
                            #print tp_n_fp[gnr_cnt+1]
                            P = counts_per_grn_cv.astype(np.float) / tp_n_fp[gnr_cnt+1]
                            P_per_gnr[gnr_cnt+1] = P[gnr_cnt+1]
                            R = counts_per_grn_cv.astype(np.float) / cv_tg_idxs[gnr_cnt+1]
                            R_per_gnr[gnr_cnt+1] = R[gnr_cnt+1]  
                            F1_per_gnr[gnr_cnt+1] = 2 * P[gnr_cnt+1] * R[gnr_cnt+1] / (P[gnr_cnt+1] + R[gnr_cnt+1]) 
                            
                        P_per_gnr[0] = precision_score(crossval_Y, predicted_Y)   
                        R_per_gnr[0] = recall_score(crossval_Y, predicted_Y) 
                        F1_per_gnr[0] = f1_score(crossval_Y, predicted_Y)  
                        
                        #Maybe Later
                        #fpr, tpr, thresholds = roc_curve(crossval_Y, predicted_Y)   
                        
                        print self.h5_res.createArray(iters_group, 'expected_Y', crossval_Y, "Expected Classes per Document (CrossValidation Set)")[:]                                         
                        print self.h5_res.createArray(iters_group, 'predicted_Y', predicted_Y, "predicted Classes per Document (CrossValidation Set)")[:]
                        print self.h5_res.createArray(iters_group, 'predicted_classes_per_iter', predicted_classes_per_iter, "Predicted Classes per Document per Iteration (CrossValidation Set)")[:]
                        print self.h5_res.createArray(iters_group, 'predicted_scores', predicted_scores, "predicted Scores per Document (CrossValidation Set)")[:]
                        print self.h5_res.createArray(iters_group, 'max_sim_scores_per_iter', max_sim_scores_per_iter, "Max Similarity Score per Document per Iteration (CrossValidation Set)")[:]                        
                        print self.h5_res.createArray(iters_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)")[:]
                        print self.h5_res.createArray(iters_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)")[:]
                        print self.h5_res.createArray(iters_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)")[:]
                        print                
                                        
               

def cosine_similarity(vector, centroid):
 
    return vector * np.transpose(centroid) / ( np.linalg.norm(vector.todense()) * np.linalg.norm(centroid) )



def hamming_similarity(vector, centroid):
 
    return 1.0 - spd.hamming(centroid, vector)



def correlation_similarity(vector, centroid):
    
    vector = vector[0]
    centroid = np.array(centroid)[0]
        
    vector_ = np.where(vector > 0, 0, 1)
    centroid_ = np.where(centroid > 0, 0, 1)
   
    s11 = np.dot(vector, centroid)
    s00 = np.dot(vector_, centroid_)
    s01 = np.dot(vector_, centroid)
    s10 = np.dot(vector,centroid_)
    
    denom = np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))
    if denom == 0.0:
        denom = 1.0
        
    return (s11*s00 - s01*s10) / denom
    
    

if __name__ == '__main__':
    
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'w')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-0.5_(forVSBagging).h5', 'w')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TEST_NOBAGG_2.h5', 'w')
    
    kfolds = 10
    vocabilary_size = [100000] #[1000,3000,10000,100000]
    iter_l = [100]
    featr_size_lst = [1000, 5000, 10000, 70000] #20000, 50000,
    sigma_threshold = 0.5
    N_Gram_size = 4
    W_N_Gram_size = 1
    
    #sparse_WNG = h2v_wcng.Html2TF(W_N_Gram_size, attrib='text', lowercase=True, valid_html=False)
    sparse_CNG = h2v_cng.Html2TF(N_Gram_size, attrib='text', lowercase=True, valid_html=False)
    
    crossV_Koppels = CrossVal_Koppels_method(sparse_CNG, CrossVal_Kopples_method_res, corpus_filepath, genres)
    
    xhtml_file_l, cls_gnr_tgs = crossV_Koppels.corpus_files_and_tags()
    
    #Just for Verifying it will return the same resaults to the new version
    def norm_func_Max( f_mtrx, len_d):
        print f_mtrx.toarray()
        print float(f_mtrx.toarray().max())
        print f_mtrx.toarray() / float(f_mtrx.toarray().max())
        return f_mtrx.toarray() / float(f_mtrx.toarray().max())


    #Cosine Similarity
    crossV_Koppels.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst,\
                                     sigma_threshold, similarity_func=cosine_similarity, sim_min_val=-1.0, norm_func=norm_func_Max)
    #Hamming Similarity
    #crossV_Koppels.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst,\
    #                                 sigma_threshold, similarity_func=correlation_similarity, sim_min_val=-1.0, norm_func=None)
    
    CrossVal_Kopples_method_res.close()


