

import numpy as np
import scipy.spatial.distance as spd
import sklearn.svm as svm



class OCSVME_Wrapped(object):
    
    def __init__(self, genres):
        self.genres_lst = genres
        self.gnrs_num = len(genres)


    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, params):
        inds_per_gnr = dict()
        inds = list()
        last_gnr_tag = 1
        
        for trn_idx in trn_idxs:
            
            if cls_gnr_tgs[trn_idx] != last_gnr_tag:
                inds_per_gnr[ self.genres_lst[last_gnr_tag - 1] ] = inds
                last_gnr_tag = cls_gnr_tgs[trn_idx]
                inds = []
            
            inds.append( trn_idx )    
        print self.genres_lst
        inds_per_gnr[ self.genres_lst[last_gnr_tag - 1] ] = inds 
    
        gnr_classes = dict()
        for g in self.genres_lst:
            
            #Create the OC-SVM Model for this genre
            gnr_classes[g] = svm.OneClassSVM(kernel='linear', nu=params['nu'], shrinking=True, cache_size=200, verbose=False)
            #print "Fit Model for ", g
            #print "Corpus_Mtrx", corpus_mtrx[inds_per_gnr[g], :].shape
            #Convert TF vectors to Binary 
            #crp_arr_bin = np.where(corpus_mtrx[inds_per_gnr[g], :].toarray() > 0, 1, 0)
            
            #Fit OC-SVM Model to Data of this genre
            gnr_classes[g].fit( corpus_mtrx[inds_per_gnr[g], 0:params['features_size']] ) # for Sparce Matrix it might require corpus_mtrx[inds_per_gnr[g], :].toarray()
        
        return gnr_classes

   
    def predict(self, *args):

        #Get Input arguments in given sequence  
        crv_idxs = args[0]                             
        corpus_mtrx = args[1]
        params = args[2]
        gnr_classes = args[3]

        #Get the part of matrices or arrays required for the model predicition phase
        crossval_X =  corpus_mtrx[ crv_idxs, 0:params['features_size'] ] 
        #EXTREMELY IMPORTANT: corpus_mtrx[X] where X=[<idx1>,<idx2>,...,<idxN>] returns ERROR HDF5 when using pytables Earray.
        #For scipy.sparse there is no such a problem. Therefore it always should be used this expression corpus_mtrx[X, : ] 

        #Get the part of matrices required for the model predicition phase
        #crossval_Y =  cls_gnr_tgs [ crv_idxs, : ]
                
        #Initialise Predicted-Classes-Arrays List 
        predicted_Y_per_gnr = list()
        predicted_dist_per_gnr = list()
        
        for cls_tag, g in enumerate(self.genres_lst):
            
            #Converting TF vectors to Binary
            #cv_arr_bin = np.where(crossval_X.toarray() > 0, 1, 0)
            
            #Getting the predictions for each Vector for this genre
            predicted_Y = gnr_classes[ g ].predict( crossval_X ) #For an one-class model, +1 or -1 is returned.
            predicted_D = gnr_classes[ g ].decision_function( crossval_X )
            #For Sparce Matricies it might require crossval_X.toarray() 

            #Assigning Genre-Class tag to Predicted_Y(s)
            predicted_Y = np.where( predicted_Y == 1, cls_tag + 1, 0 )
            
            #Keeping the prediction per genre 
            predicted_Y_per_gnr.append( predicted_Y )
            predicted_dist_per_gnr.append( predicted_D.reshape( predicted_D.shape[0] ) ) 
            
        #Converting it to Array before returning
        predicted_Y_per_gnr = np.vstack( predicted_Y_per_gnr )
        predicted_dist_per_gnr = np.vstack( predicted_dist_per_gnr )

        #Finding index of the Max Positive distancies from the Ensembles Predicted distance Array/Matrix
        max_dist_idxs =  np.argmax(predicted_dist_per_gnr, axis=0)

        #Keeping the Max Positive distance form Predicted distancies Array/Matrix and the respected Predicted Ys 
        predicted_Y_per_gnr = predicted_Y_per_gnr[ max_dist_idxs ]
        predicted_dist_per_gnr = predicted_dist_per_gnr[ max_dist_idxs ]

        #Selecting tha maximum score per column, i.e., per genre-learner for each document.
        max_scores_idxs = np.argmax( predicted_dist_per_gnr, axis=0 )
        
        #Getting the maximum scores as selected above.
        predicted_scores = np.choose(max_scores_idxs, predicted_dist_per_gnr) 
        
        #Getting the Y's respectively to maximum scores as selected above.
        predicted_Y = np.choose(max_scores_idxs, predicted_Y_per_gnr) 
    
        return (predicted_Y, predicted_scores, predicted_Y_per_gnr, predicted_dist_per_gnr) 


    def eval(self, *args):

        #Get Input arguments in given sequence
        trn_idxs = args[0]  
        crv_idxs = args[1]                             
        corpus_mtrx = args[2]
        cls_gnr_tgs = args[3]
        vocab_index_dct = args[4] #tid
        params = args[5]
        
        #Build Genre Classes given the training vectors
        gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs, params)

        #Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
        results = self.predict(crv_idxs, corpus_mtrx, params, gnr_classes)

        #Expected Results for the ParamGridCrossValBase class in paramgridcrossval module
        predicted_Y = results[0]
        predicted_scores = results[1]
        model_specific_d = {\
            'predicted_Y_per_gnr' : results[2],\
            'predicted_dist_per_gnr' : results[3]
        }

        #Return results as expected form ParamGridCrossValBase class
        return predicted_Y, predicted_scores, model_specific_d
        
 

