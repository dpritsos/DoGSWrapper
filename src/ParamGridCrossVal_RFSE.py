"""     """

import sys
#sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')

import json
import os
import cPickle as pickle

import numpy as np
import tables as tb

import scipy.sparse as ssp
import scipy.spatial.distance as spd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation
from sklearn import grid_search #.IterGrid

import html2vect.sparse.wngrams as h2v_wcng
import html2vect.sparse.cngrams as h2v_cng



class ParamGridCrossValBase(object):
    
    def __init__(self, TF_TT, h5_res, corpus_path, genres, voc_path=None):
        self.TF_TT = TF_TT
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        self.h5_res = h5_res
        self.crps_voc_path = voc_path


    def corpus_files_and_tags(self):
        #Creating a Group for this Vocabulary size in h5 file under this k-fold
        try:
            print "LOADING HTML FILE LIST FROM H5 File" 
            html_file_l = self.h5_res.getNode('/', 'HTML_File_List')
            cls_gnr_tgs = self.h5_res.getNode('/', 'Class_Genres_Tags')

        except:
            print "CREATING"
            html_file_l = list()
            cls_gnr_tgs = list()
            for i, g in enumerate(self.genres_lst):
                gnrs_file_lst = self.TF_TT.file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
                
                html_file_l.extend( gnrs_file_lst )
                
                cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )

            html_file_l = self.h5_res.createArray('/', 'HTML_File_List', np.array(html_file_l),\
                "HTML File List as founded in the Ext4 file system by python built-it os (python 2.7.x) lib" )

            cls_gnr_tgs = self.h5_res.createArray('/', 'Class_Genres_Tags', np.array(cls_gnr_tgs),\
                "Assigned Genre Tags to files list Array" )
    
        return (html_file_l.read(), cls_gnr_tgs.read())
    
                      
    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, bagging_param):
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
            
            #######
            shuffled_train_idxs = np.random.permutation( inds_per_gnr[g] )
            #print shuffled_train_idxs
            #keep bagging_parram percent
            bg_trn_ptg = int( np.trunc( shuffled_train_idxs.size * bagging_param ) )
            #print bg_trn_ptg
            bag_idxs = shuffled_train_idxs[0:bg_trn_ptg]
            #print bag_idxs
            ######
        
            #Merge All Term-Frequency Dictionaries created by the Raw Texts
            gnr_classes[g] = corpus_mtrx[bag_idxs, :].mean(axis=0)
        
        return gnr_classes

    
    def predict(self, *args):

        #Put arguments into classes
        bagging_param = args[0]
        crossval_X =  args[1]  
        crossval_Y =  args[2] 
        vocab_index_dct = args[3] 
        featrs_size =  args[4] 
        similarity_func = args[5] 
        sim_min_value =  args[6] 
        iters =  args[7] 
        sigma_threshold = args[8]
        trn_idxs = args[9]  
        corpus_mtrx = args[10]  
        cls_gnr_tgs = args[11]  
            
        max_sim_scores_per_iter = np.zeros((iters, crossval_X.shape[0]))
        predicted_classes_per_iter = np.zeros((iters, crossval_X.shape[0]))
                    
        #Measure similarity for iters iterations i.e. for iters different feature subspaces Randomly selected 
        for I in range(iters):

            #print "Construct classes"
            #Construct Genres Class Vectors form Training Set
            gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs, bagging_param)
            
            #Randomly select some of the available features
            shuffled_vocabilary_idxs = np.random.permutation( np.array(vocab_index_dct.values()) ) 
            features_subspace = shuffled_vocabilary_idxs[0:featrs_size]
            
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
        
    
    def evaluate(self, *args):

        html_file_l = args[0]
        cls_gnr_tgs = args[1]
        norm_func = args[2]
        similarity_func = args[3]
        sim_min_val = args[4]
        params_range = args[5]
        encoding = args[6]

        #Create CrossVal Folds
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, len(params_range['kfolds']), indices=True)
        
        for k, (trn, crv) in enumerate(KF):

            voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.vtf'
            pkl_voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.pkl'
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'

            #Save K-Fold Cross-Validation corpus vector selection-indecies if does not exists
            if not os.path.exists(trn_filename) or not os.path.exists(crv_filename):
              
                #Save Trainging Indeces
                print "Saving Training Indices for k-fold=", k
                with open(trn_filename, 'w') as f:
                    json.dump( list(trn), f, encoding=encoding)

                #Save Cross-validation Indeces
                print "Saving Cross-validation Indices for k-fold=", k
                with open(crv_filename, 'w') as f:
                    json.dump( list(crv), f, encoding=encoding)               

            #Load or Create K-fold Cross-Validation Vocabulary for each fold
            if not os.path.exists(voc_filename) or not os.path.exists(pkl_voc_filename):
         
                #Creating Vocabulary
                print "Creating Vocabulary for k-fold=",k 
                print  html_file_l[32]
                tf_d = self.TF_TT.build_vocabulary( list( html_file_l[trn] ), encoding=encoding, error_handling='replace' )

                #Saving Vocabulary
                print "Saving Vocabulary"
                with open(pkl_voc_filename, 'w') as f:
                    pickle.dump(tf_d, f)

                with open(voc_filename, 'w') as f:
                    json.dump(tf_d, f, encoding=encoding)

        #Starting Parameters Grid Search 
        for params in grid_search.IterGrid(params_range):

            #Prevent execution of this loop in case feature_size is smaller than Vocabulary size
            if params['features_size'] > params['vocab_size']:
                print "SKIPPEd Params: ", params
                continue                    

            print "Params: ", params
            #bs = cross_validation.Bootstrap(9, random_state=0)
            #Set Experiment Parameters
            k = params['kfolds']
            iters = params['training_iter']
            vocab_size = params['vocab_size']
            featrs_size = params['features_size']
            sigma_threshold = params['threshold']
            bagging_param = params['bagging_param']

            #Creating a Group for this Vocabulary size in h5 file under this k-fold
            try:
                vocab_size_group = self.h5_res.getNode('/', 'Vocab'+str(vocab_size))    
            except:
                vocab_size_group = self.h5_res.createGroup('/', 'Vocab'+str(vocab_size),\
                                "Vocabulary actual size group of Results Arrays for this K-fold" )

            #Creating a Group for this features size in h5 file under this k-fold
            try:
                feat_num_group = self.h5_res.getNode(vocab_size_group, 'Feat'+str(featrs_size))    
            except:
                feat_num_group = self.h5_res.createGroup(vocab_size_group, 'Feat'+str(featrs_size),\
                                "Features Number group of Results Arrays for this K-fold" )
            
            #Creating a Group for this number of iterations in h5 file under this features number under this k-fold
            try:
                iters_group = self.h5_res.getNode(feat_num_group, 'Iters'+str(iters))
            except:
                iters_group = self.h5_res.createGroup(feat_num_group, 'Iters'+str(iters),\
                            "Number of Iterations (for statistical prediction) group of Results Arrays for this K-fold" )

            #Creating a Group for this Sigma_thershold in h5 file under this features number under this k-fold
            try:
                sigma_group = self.h5_res.getNode(iters_group, 'Sigma'+str(sigma_threshold).replace('.',''))
            except:
                sigma_group = self.h5_res.createGroup(iters_group, 'Sigma'+str(sigma_threshold).replace('.',''),\
                            "<Comment>" )

            #Creating a Group for this Bagging_Param in h5 file under this features number under this k-fold
            try:
                bagg_group = self.h5_res.getNode(sigma_group, 'Bagg'+str(bagging_param).replace('.',''))
            except:
                bagg_group = self.h5_res.createGroup(sigma_group, 'Bagg'+str(bagging_param).replace('.',''),\
                            "<Comment>" )

            #Creating a Group for this k-fold in h5 file
            try:
                kfld_group = self.h5_res.getNode(bagg_group, 'KFold'+str(k))
            except:
                kfld_group = self.h5_res.createGroup(bagg_group, 'KFold'+str(k), "K-Fold group of Results Arrays")

            #Loading Vocavulary
            print "Loadinging VOCABULARY for k-fold=",k
            with open(pkl_voc_filename, 'r') as f:
                tf_d = pickle.load(f)
            
            #Get the Vocabuliary keeping all the terms with same freq to the last feature of the reqested size
            resized_tf_d = self.TF_TT.tfdtools.keep_atleast(tf_d, vocab_size) 

            #Create The Terms-Index Vocabulary that is shorted by Frequency descending order
            tid = self.TF_TT.tfdtools.tf2tidx( resized_tf_d )
            print tid.items()[0:5]

            #keep as pytables group attribute the actual Vocabulary size
            if k == 0:
                vocab_size_group._v_attrs.real_voc_size_per_kfold = [len(resized_tf_d)]
            else:
                vocab_size_group._v_attrs.real_voc_size_per_kfold += [len(resized_tf_d)]

            #Load or Crreate the Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
            corpus_mtrx_fname = self.crps_voc_path+'/kfold_VocSize_'+str(k)+str(vocab_size)+'.pkl'

            if os.path.exists(corpus_mtrx_fname):
                print "Loading Sparse TF Matrix for CrossValidation for K-fold=", k, " and Vocabulary size=", vocab_size
                #Loading Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
                with open(corpus_mtrx_fname, 'r') as f:
                    corpus_mtrx = pickle.load(f)

            else:
                print "Creating Sparse TF Matrix (for CrossValidation) for K-fold=", k, " and Vocabulary size=", vocab_size
                #Creating TF Vectors Sparse Matrix
                corpus_mtrx = self.TF_TT.from_files(list( html_file_l ), tid_dictionary=tid, norm_func=norm_func,\
                                                    encoding='utf8', error_handling='replace' )[0]

                #Saving TF Vecrors Matrix
                print "Saving Sparse TF Matrix (for CrossValidation)"
                with open(corpus_mtrx_fname, 'w') as f:
                    pickle.dump(corpus_mtrx, f)
                
            #Load Training Indeces 
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            print "Loading Training Indices for k-fold=", k
            with open(trn_filename, 'r') as f:
                trn_idxs = np.array( json.load(f, encoding=encoding) )

            """ 
            It has been moved in predict() function for enabling Bagging with ease
            print "Construct classes"
            #Construct Genres Class Vectors form Training Set
            gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx[0], cls_gnr_tgs)

            """

            #Load Cross-validation Indeces
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'
            print "Loading Cross-validation Indices for k-fold=", k
            with open(crv_filename, 'r') as f:
                crv_idxs = np.array( json.load(f, encoding=encoding) )

            #Select Cross Validation Set
            crossval_Y = cls_gnr_tgs[ crv_idxs ]
            mtrx = corpus_mtrx
            crossval_X = mtrx[crv_idxs, :]
                
            print "EVALUATE"
            #Evaluating Classification Method
            predicted_Y,\
            predicted_scores,\
            max_sim_scores_per_iter,\
            predicted_classes_per_iter = self.predict(\
                                            bagging_param,\
                                            crossval_X, crossval_Y,\
                                            tid, featrs_size,\
                                            similarity_func, sim_min_val,\
                                            iters, sigma_threshold,\
                                            trn_idxs, corpus_mtrx, cls_gnr_tgs,\
                                         ) 
            
            #Calculating Scores Precision, Recall and F1 Statistic
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
            
            #Saving results
            print self.h5_res.createArray(kfld_group, 'expected_Y', crossval_Y, "Expected Classes per Document (CrossValidation Set)")[:]                                         
            print self.h5_res.createArray(kfld_group, 'predicted_Y', predicted_Y, "predicted Classes per Document (CrossValidation Set)")[:]
            print self.h5_res.createArray(kfld_group, 'predicted_classes_per_iter', predicted_classes_per_iter, "Predicted Classes per Document per Iteration (CrossValidation Set)")[:]
            print self.h5_res.createArray(kfld_group, 'predicted_scores', predicted_scores, "predicted Scores per Document (CrossValidation Set)")[:]
            print self.h5_res.createArray(kfld_group, 'max_sim_scores_per_iter', max_sim_scores_per_iter, "Max Similarity Score per Document per Iteration (CrossValidation Set)")[:]                        
            print self.h5_res.createArray(kfld_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)")[:]
            print self.h5_res.createArray(kfld_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)")[:]
            print self.h5_res.createArray(kfld_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)")[:]
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
    #kfolds_vocs_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/Kfolds_Vocabularies_4grams"
    kfolds_vocs_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Kfolds_Vocs_Inds_4Grams"
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'w')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TEST_NOBAGG.h5.h5', 'w')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Char4Grams-Koppels-Bagging_method_kfolds-10_GridSearch_TEST.h5', 'w')
    

    params_range = {
        'kfolds' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'vocab_size' : [100000],
        'features_size' : [1000, 5000, 10000, 70000],
        'training_iter' : [100],
        'threshold' : [0.5],
        'bagging_param' : [0.66],
    } 

    N_Gram_size = 4
    W_N_Gram_size = 1
    
    #sparse_WNG = h2v_wcng.Html2TF(W_N_Gram_size, attrib='text', lowercase=True, valid_html=False)
    sparse_CNG = h2v_cng.Html2TF(N_Gram_size, attrib='text', lowercase=True, valid_html=False)
    
    crossV_Koppels = ParamGridCrossValBase( sparse_CNG, CrossVal_Kopples_method_res, corpus_filepath,\
                                            genres, kfolds_vocs_filepath )
    
    html_file_l, cls_gnr_tgs = crossV_Koppels.corpus_files_and_tags()

    crossV_Koppels.evaluate(html_file_l, cls_gnr_tgs, None, cosine_similarity, -1.0, params_range, 'utf-8')
    #Hamming Similarity
    #crossV_Koppels.evaluate(xhtml_file_l, cls_gnr_tgs, kfolds, vocabilary_size, iter_l, featr_size_lst,\
    #                                 sigma_threshold, similarity_func=correlation_similarity, sim_min_val=-1.0, norm_func=None)
    
    CrossVal_Kopples_method_res.close()