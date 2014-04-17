
import sys
sys.path.append('../../html2vectors/src')

import json
import os
import cPickle as pickle

import numpy as np
import tables as tb
import numexpr as ne

import scipy.sparse as ssp
import scipy.spatial.distance as spd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation
import param_combs
from html2vect.utils import tfdutils
from html2vect.base.io.basefilehandlers import file_list_frmpaths



class ParamGridCrossValBase(object):
    
    def __init__(self, ML_Model, TF_TT, h5_res, genres, corpus_path, voc_path):
        self.model = ML_Model
        self.TF_TT = TF_TT
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        self.h5_res = h5_res
        self.crps_voc_path = voc_path


    def calculate_p_r_f1(self, crossval_Y, predicted_Y):

        #Calculating Scores Precision, Recall and F1 Statistic
        #print np.histogram(crossval_Y, bins=np.arange(self.gnrs_num+2))
        #print np.histogram(predicted_Y.astype(np.int), bins=np.arange(self.gnrs_num+2))
        
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

        #Check this how it is working with Multiclass case [ 1,.., 2,.., 3,.., x,... ]
        P_per_gnr[0] = -1 #precision_score(crossval_Y, predicted_Y)   
        R_per_gnr[0] = -1 #recall_score(crossval_Y, predicted_Y) 
        F1_per_gnr[0] = -1 #f1_score(crossval_Y, predicted_Y)  

        return (P_per_gnr, R_per_gnr, F1_per_gnr)


    def corpus_files_and_tags(self, gnr_file_idx=None, iidx=None):

        corpus_files_lst_path = self.crps_voc_path+'/Corpus_filename_shorted.lst'
        corpus_tags_lst_path = self.crps_voc_path+'/Corpus_tags_shorted.lst'

        if os.path.exists(corpus_files_lst_path) and  os.path.exists(corpus_files_lst_path):
            
            print "Loading HTML Filenames and Classes Tags Lists" 
            
            #Load Filename and classes Tags lists
            with open(corpus_files_lst_path, 'r') as f:
                html_file_l = json.load(f, encoding='utf-8')

            with open(corpus_tags_lst_path, 'r') as f:
                cls_gnr_tgs = json.load(f, encoding='utf-8')
            
        else:
            
            print "Creating and Saving HTML Filenames and Classes Tags Lists"
            
            html_file_l = list()
            cls_gnr_tgs = list()
            if gnr_file_idx and iidx:
                #Get the index file showing the respective genre of each file of the given corpus
                with open(self.corpus_path+gnr_file_idx, 'r') as f:
                    #Getting the list of tuples (index, file) using the index-of-index list (iidx) argument
                    #iidx[0] == the splittig character, iidx[1] == the field that contains the filename , 
                    #iidx[2] == the field that containd the genre of the file
                    gnrs_file_lst = [ (line.split( iidx[0] )[ iidx[2] ], line.split( iidx[0] )[ iidx[1] ]) for line in f ]
                
                #Sort the above tuples list based on genre 
                sorted(gnrs_file_lst, key=lambda gnrs_file_lst: gnrs_file_lst[0])
                
                #just for debugging remove it right after
                #print gnrs_file_lst

                #Get the filenames as sorted above
                html_file_l = [ element[1] for element in gnrs_file_lst ]
                
                #Build the class-genre-tag list by assigning as a tag the index number of the list of genre given as argument to this
                #class, i.e. ParamGridCrossValBase()
                cls_gnr_tgs = [ genres_lst.index( element[0] ) for element[1] in gnrs_file_lst ]

            elif gnr_file_idx == None  and iidx == None:
                #Get the list of Genre argument as given to this Class and build html-file-list and class-genres-tags list
                for i, g in enumerate(self.genres_lst):
                    #Get all files located to the genre's path 'g'
                    gnrs_file_lst = file_list_frmpaths(self.corpus_path, [ str( g + "/html/" ) ] )
                    
                    #Extends the list of html files with the set of files form genre 'g'
                    html_file_l.extend( gnrs_file_lst )
                    
                    #Extends the list of html files with the set of class tag form genre 'g', i.e. the index of the 
                    #genre's list given as argument to this class ( ParamGridCrossValBase() ).
                    cls_gnr_tgs.extend( [i+1]*len(gnrs_file_lst) )

            else:
                raise Exception("Both Genre-of-Files-Index and Index-of-Index arguments should be given or 'None' of them")

            #Saving Filename and classes Tags lists
            with open(corpus_files_lst_path, 'w') as f:
                #HTML File List as founded in the Ext4 file system by python built-it os (python 2.7.x) lib
                json.dump(html_file_l, f, encoding='utf-8')

            with open(corpus_tags_lst_path, 'w') as f:
                #Assigned Genre Tags to files list Array
                json.dump(cls_gnr_tgs, f, encoding='utf-8')
    
        return (np.array(html_file_l), np.array(cls_gnr_tgs))


    def corpus_matrix(self, k, vocab_size_group, vocab_size, html_file_l, tid, norm_func):

        #Load or Crreate the Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
        corpus_mtrx_fname = self.crps_voc_path+'/kfold_CorpusMatrix_'+str(k)+str(vocab_size)+'.pkl'

        if os.path.exists(corpus_mtrx_fname):
            print "Loading Sparse TF Matrix for CrossValidation for K-fold=", k, " and Vocabulary size=", vocab_size
            #Loading Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
            with open(corpus_mtrx_fname, 'r') as f:
                corpus_mtrx = pickle.load(f)

        else:
            print "Creating Sparse TF Matrix (for CrossValidation) for K-fold=", k, " and Vocabulary size=", vocab_size
            #Creating TF Vectors Sparse Matrix
            corpus_mtrx = self.TF_TT.from_files(list( html_file_l ), tid_dictionary=tid, norm_func=norm_func,\
                                                encoding='utf8', error_handling='replace' )[0] #<--- Be carefull with zero index

            #Save the Webpages term counts (Char N-grans or Word N-Grams)
            kfld_group = self.h5_res.createGroup(vocab_size_group, 'KFold'+str(k))
            docs_term_counts = self.h5_res.createArray(kfld_group, 'docs_term_counts', np.sum(corpus_mtrx.toarray(), axis=1))

            #Perform default (division by max value) normalisation for corpus matrix 'corpus_mtrx'
            #Should I perform Standarisation/Normalisation by substracting mean value from vector variables?
            print "Normalising"

            #Getting the Maximum frequency for every document.
            max_vals = np.max(corpus_mtrx.todense(), axis=1)

            #For Documents with zero terms. This case occurs when a sub-Vocabulary is used for the experiment.
            max_vals[ np.where(max_vals == 0) ] = 1

            #Normalising
            corpus_mtrx = ssp.csr_matrix( corpus_mtrx.todense() / max_vals )

            #Saving TF Vecrors Matrix
            print "Saving Sparse Normalized TF Matrix (for CrossValidation)"
            with open(corpus_mtrx_fname, 'w') as f:
                pickle.dump(corpus_mtrx, f)

        return (corpus_mtrx, None)
    

    def get_test_only_idxs(self, cls_gnr_tgs, test_only_tgs):

        if len(test_only_tgs) > 1:
            valid_tgs_lst = range(test_only_tgs[0], len(test_only_tgs), 1)

            if valid_tgs_lst != test_only_tgs:
                raise Exception("Invalid test-only-tags sequence: Only a numerical sequence with step +1 increment is valid")

            inv_cls_gnr_tgs = cls_gnr_tgs[::-1]
            if inv_cls_gnr_tgs[0:len(test_only_tgs)] != test_only_tgs[::-1]:
                raise Exception("Invalid test-only-tags sequence: Only the most last tags of the cls_gnr_tgs can be used as only-for-test")            
        
        new_cls_gnr_tgs = list()
        test_only_idxs = list()

        for i, tag in enumerate(cls_gnr_tgs):

            if tag in test_only_tgs:
                test_only_idxs.append(i)
            else:
                new_cls_gnr_tgs.append(tag)

        return ( new_cls_gnr_tgs, test_only_idxs )
        

    def evaluate(self, *args):
        """ Call prototyping: evaluate(html_file_l, cls_gnr_tgs, None, params_range, 'utf-8') """

        html_file_l = args[0]
        cls_gnr_tgs = args[1]
        test_only_tgs = args[2]
        norm_func = args[3]
        params_range = args[4]
        encoding = args[5]

        #Separationg the "Test-Only" tags for the rest
        if test_only_tgs:
            cls_gnr_tgs, test_only_idxs = self.get_test_only_idxs(cls_gnr_tgs, test_only_tgs)

        #Create CrossVal Folds
        KF = cross_validation.StratifiedKFold(cls_gnr_tgs, len(params_range['kfolds']), indices=True)

        #Appending the test only tags the "Test-Only" tags for the rest - this works only for the last most in genres list() 
        #Since it is required to build the "crv" and "trn" indecies first and then load them for using then it is required 
        #to create the Corssvalidations sets form the rest of the tags set and then append them. Therefor for selecting the proper
        #tags it is required to be appended back to the cls_gnr_tgs but this time with Zero(0) class tag. 
        #Need to be simplafied!
        if test_only_tgs:
            cls_gnr_tgs = np.hstack( (np.array(cls_gnr_tgs), np.zeros(len(test_only_idxs), dtype=np.int32)) )
        
        for k, (trn, crv) in enumerate(KF):

            voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.vtf'
            pkl_voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.pkl'
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'

            #Save K-Fold Cross-Validation corpus vector selection-indecies if does not exists
            #Create K-fold Cross-Validation Vocabulary for each fold
            #Stratified Indecies and respective Vocabularies should be syncornisied therefore 
            #there saving-files should be created all together if one or more are missing
            if not os.path.exists(trn_filename) or not os.path.exists(crv_filename) or not os.path.exists(voc_filename) or not os.path.exists(pkl_voc_filename):
                
                #Save Trainging Indeces
                print "Saving Training Indices for k-fold=", k
                with open(trn_filename, 'w') as f:
                    json.dump( list(trn), f, encoding=encoding)

                #Save Cross-validation Indeces
                print "Saving Cross-validation Indices for k-fold=", k
                with open(crv_filename, 'w') as f:
                    if test_only_tgs:
                        json.dump( list(crv) + test_only_idxs, f, encoding=encoding)   
                    else:
                        json.dump( list(crv), f, encoding=encoding)               
                                    
         
                #Creating Vocabulary
                print "Creating Vocabulary for k-fold=",k
                tf_d = self.TF_TT.build_vocabulary( list( html_file_l[trn] ), encoding='utf-8', error_handling='replace' )
                

                #Saving Vocabulary
                print "Saving Vocabulary"
                with open(pkl_voc_filename, 'w') as f:
                    pickle.dump(tf_d, f)

                with open(voc_filename, 'w') as f:
                    json.dump(tf_d, f, encoding=encoding)

        #Starting Parameters Grid Search 
        for gci, params in enumerate(  param_combs.ParamGridIter(params_range) ):

            #Show how many Gric Search Parameter combinations are remaning
            print "Param Grid Counts:", gci+1

            #Prevent execution of this loop in case feature_size is smaller than Vocabulary size
            if params['features_size'] > params['vocab_size']:
                print "SKIPPED Params: ", params
                continue                    

            print "Params: ", params
            #bs = cross_validation.Bootstrap(9, random_state=0)
            #Set Experiment Parameters
            k = params['kfolds']
            vocab_size = params['vocab_size']
            featrs_size = params['features_size']

            #Creating a Group for this Vocabulary size in h5 file under this k-fold
            try:
                vocab_size_group = self.h5_res.getNode('/', 'vocab_size'+str(vocab_size))    
            except:
                vocab_size_group = self.h5_res.createGroup('/', 'vocab_size'+str(vocab_size),\
                                "Vocabulary actual size group of Results Arrays for this K-fold" )
                                   
            #Creating a Group for this features size in h5 file under this k-fold
            try:
                feat_num_group = self.h5_res.getNode(vocab_size_group, 'features_size'+str(featrs_size))    
            except:
                feat_num_group = self.h5_res.createGroup(vocab_size_group, 'features_size'+str(featrs_size),\
                                "Features Number group of Results Arrays for this K-fold" )
            
            ###Create the group sequence respectively to the models parameters:

            #Assigne Feature number group to next_group parameter for initializing the loop
            next_group = feat_num_group

            #Start the loop of creating of getting group nodes in respect to model parameters
            for pname, pvalue in params.items():
                if pname not in ['kfolds', 'vocab_size', 'features_size']:           
                    try:
                        next_group = self.h5_res.getNode(next_group, pname+str(pvalue).replace('.',''))
                    except:
                        next_group = self.h5_res.createGroup(next_group, pname+str(pvalue).replace('.',''), "<Comment>" )   

            ###END- Group creation sequence 

            #Creating a Group for this k-fold in h5 file
            try:
                kfld_group = self.h5_res.getNode(next_group, 'KFold'+str(k))
            except:
                kfld_group = self.h5_res.createGroup(next_group, 'KFold'+str(k), "K-Fold group of Results Arrays")

            #Loading Vocabulary
            print "Loadinging VOCABULARY for k-fold=",k
            with open(pkl_voc_filename, 'r') as f:
                tf_d = pickle.load(f)
            
            #Get the Vocabuliary keeping all the terms with same freq to the last feature of the reqested size
            resized_tf_d = tfdutils.keep_atleast(tf_d, vocab_size) 

            #Saving the real Vocabulary sizes for this experiment (i.e. this text representation, etc.)
            #keep it as pytables group attribute the actual Vocabulary size
            vocab_size_group._v_attrs.real_voc_size = [(k, len(resized_tf_d))]

            #Create The Terms-Index Vocabulary that is shorted by Frequency descending order
            tid = tfdutils.tf2tidx( resized_tf_d )
            #print tid.items()[0:5]

            #Load or Create the Coprus Matrix/Array for this combination or kfold and vocabulary_size
            corpus_mtrx, corpus_file = self.corpus_matrix(k, vocab_size_group, vocab_size, html_file_l, tid, norm_func)

            #Load Training Indeces 
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            print "Loading Training Indices for k-fold=", k
            with open(trn_filename, 'r') as f:
                trn_idxs = np.array( json.load(f, encoding=encoding) )

            #Load Cross-validation Indeces
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'
            print "Loading Cross-validation Indices for k-fold=", k
            with open(crv_filename, 'r') as f:
                crv_idxs = np.array( json.load(f, encoding=encoding) )

            print "EVALUATE"
            #Evaluating Classification Method
            predicted_Y, predicted_scores,\
            model_specific_d = self.model.eval(\
                                    trn_idxs, crv_idxs,\
                                    corpus_mtrx, cls_gnr_tgs, tid,\
                                    params\
                                ) 

            #Select Cross Validation Set
            #print cls_gnr_tgs
            #print crv_idxs
            crossval_Y = cls_gnr_tgs[ crv_idxs ]
                
            P_per_gnr, R_per_gnr, F1_per_gnr = self.calculate_p_r_f1(crossval_Y, predicted_Y)
                        
            #Saving results
            self.h5_res.createArray(kfld_group, 'expected_Y', crossval_Y, "Expected Classes per Document (CrossValidation Set)")[:]                                         
            self.h5_res.createArray(kfld_group, 'predicted_Y', predicted_Y, "predicted Classes per Document (CrossValidation Set)")[:]
            self.h5_res.createArray(kfld_group, 'predicted_scores', predicted_scores, "predicted Scores per Document (CrossValidation Set)")[:]
            self.h5_res.createArray(kfld_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)")[:]
            self.h5_res.createArray(kfld_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)")[:]
            self.h5_res.createArray(kfld_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)")[:]

            print 
            
            if  model_specific_d:
                for name, value in model_specific_d.items():
                    self.h5_res.createArray(kfld_group, name, value, "<Comment>")[:]             
        
            #Closing corpus file if any. Originaly for closing hd5 files
            if corpus_file:
                corpus_file.close()
        
        #Return Resuls H5 File handler class
        return self.h5_res                                    



class ParamGridCrossValTables(ParamGridCrossValBase):

    def __init__(self, ML_Model, TF_TT, h5_res, genres, corpus_path, voc_path):

        #Passing the argument to the Super-Class
        super(ParamGridCrossValTables, self).__init__(ML_Model, TF_TT, h5_res, genres, corpus_path, voc_path)    


    def corpus_matrix(self, k, vocab_size_group, vocab_size, html_file_l, tid, norm_func):

        #Load or Crreate the Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
        corpus_mtrx_fname = self.crps_voc_path+'/kfold_CorpusMatrix_'+str(k)+str(vocab_size)+'.h5'

        if os.path.exists(corpus_mtrx_fname):
            print "Loading pyTables TF EArray for CrossValidation for K-fold=", k, " and Vocabulary size=", vocab_size
            #Loading Coprus pyTables TF EArray for this combination or kfold and vocabulary_size
            h5f = tb.openFile(corpus_mtrx_fname, 'r+')
            corpus_mtrx = h5f.getNode('/',  'corpus_earray') #h5f.root.corpus_earray 

        else:
            print "Creating pyTables TF EArray (for CrossValidation) for K-fold=", k, " and Vocabulary size=", vocab_size
            
            #Creating pyTables TF EArray
            corpus_mtrx, h5f = self.TF_TT.from_files(list( html_file_l ), corpus_mtrx_fname, tid_dictionary=tid, norm_func=norm_func,\
                                                encoding='utf8', error_handling='replace' )[0:2] #<--- Getting only 2 of the 3 returend values
            
            #Save the Webpages term counts (Char N-grans or Word N-Grams)
            kfld_group = self.h5_res.createGroup(vocab_size_group, 'KFold'+str(k))
            docs_term_counts = self.h5_res.createArray(kfld_group, 'docs_term_counts', np.sum(corpus_mtrx, axis=1))

            #Performing default (division by max value) normalisation for corpus matrix 'corpus_mtrx'
            #Should I perform Standarisation/Normalisation by substracting mean value from vector variables?            
            #The following way of normalisatoin supposed to be executed faster and 
            #it requires much less memory because it prevents the use of 
            #indermidate array which it is required in 'c = c / c.max()' operation
            print "Normalising"   
            max_col_arr = np.max(corpus_mtrx, axis=1)[:, np.newaxis]

            #For Documents with zero terms. This case occurs when a sub-Vocabulary is used for the experiment.
            max_col_arr[ np.where(max_col_arr == 0) ] = 1

            for i, (row, max_val) in enumerate( zip(corpus_mtrx.iterrows(), max_col_arr) ):
                corpus_mtrx[i] = row / max_val

        return (corpus_mtrx, h5f)
