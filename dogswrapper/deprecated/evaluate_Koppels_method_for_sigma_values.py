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
        
class Eval_CrossVal_Koppels_method(object):
    
    def __init__(self, h5_res, corpus_path, genres):
        self.corpus_path = corpus_path
        self.genres_lst = genres
        self.gnrs_num = len(genres)
        self.h5_res = h5_res
                                        
    def predict(self, expected_Y, predicted_Y_iter, iters, sigma_threshold):
        
        predicted_Y = np.zeros((expected_Y.shape[0]), dtype=np.float)
        predicted_scores = np.zeros((expected_Y.shape[0]), dtype=np.float)
        
        for i_prd_cls, prd_cls in enumerate(predicted_Y_iter.transpose()):
            genres_occs = np.histogram( prd_cls.astype(np.int), bins=np.arange(self.gnrs_num+2))[0] #One Bin per Genre plus one i.e the first to be always zero
            #print genres_occs
            genres_probs = genres_occs.astype(np.float) / np.float(iters)
            #print genres_probs
            if np.max(genres_probs) >= sigma_threshold:
                predicted_Y[i_prd_cls] = np.argmax( genres_probs )
                predicted_scores[i_prd_cls] = np.max( genres_probs )
        
        return predicted_Y, predicted_scores      
        
    
    def evaluate(self, kfolds, iter_l, featr_size_lst, simga_threshold_list):
        
        for sigma_threshold in simga_threshold_list:
            
            for featr_size in featr_size_lst:
                 
                for iters in iter_l:
                    
                    for k in range(kfolds):
                        
                        #Get the predicted classes per iteration and the expected classes 
                        predicted_Y_iter = self.h5_res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters'+str(iters), name='predicted_classes_per_iter' )
                        predicted_Y_iter = predicted_Y_iter.read()
                        expected_Y = self.h5_res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters'+str(iters), name='expected_Y' )
                        expected_Y = expected_Y.read()
                        
                        #Get the Predicted Classes (predicted_Y) and Scores for this Sigma Threshold
                        predicted_Y, predicted_scores = self.predict(expected_Y, predicted_Y_iter, iters, sigma_threshold)        
                        
                        #Store the Results for this Sigma
                        print np.histogram(expected_Y, bins=np.arange(self.gnrs_num+2))
                        print np.histogram(predicted_Y.astype(np.int), bins=np.arange(self.gnrs_num+2))
                        
                        cv_tg_idxs = np.array( np.histogram(expected_Y, bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
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
                            F1_per_gnr[gnr_cnt+1] = 2.0 * P[gnr_cnt+1] * R[gnr_cnt+1] / (P[gnr_cnt+1] + R[gnr_cnt+1]) 
                            
                        P_per_gnr[0] = precision_score(expected_Y, predicted_Y)   
                        R_per_gnr[0] = recall_score(expected_Y, predicted_Y) 
                        F1_per_gnr[0] = f1_score(expected_Y, predicted_Y)  
                        
                        #Maybe Later
                        #fpr, tpr, thresholds = roc_curve(expected_Y, predicted_Y)   
                        
                        #Creating a Group for this Sigma
                        grp = self.h5_res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Iters'+str(iters))
                        storage_grp = self.h5_res.createGroup(grp, 'Sigma'+str(sigma_threshold)) 
                        
                        print self.h5_res.createArray(storage_grp, 'predicted_Y', predicted_Y, "predicted Classes per Document (CrossValidation Set)")[:]
                        print self.h5_res.createArray(storage_grp, 'predicted_scores', predicted_scores, "predicted Scores per Document (CrossValidation Set)")[:]                       
                        print self.h5_res.createArray(storage_grp, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)")[:]
                        print self.h5_res.createArray(storage_grp, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)")[:]
                        print self.h5_res.createArray(storage_grp, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)")[:]
                        print                
                                     
                                        
               
if __name__ == '__main__':
    
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    #genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop" ]
    #crp_crssvl_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santini_TT-Words_TM-Derivative(+-).h5', 'w')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Char4Grams-Koppels_method_kfolds-10_SigmaThreshold-None_nrmMAX.h5', 'a')
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI04_TT-Words-Koppels_method_kfolds-10_SigmaThreshold-None.h5', 'a')
    
    kfolds = 10
    iter_l = [100]
    featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000] 
    sigma_threshold_list = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    eval_crossV_Koppels = Eval_CrossVal_Koppels_method(CrossVal_Kopples_method_res, corpus_filepath, genres)
    
    eval_crossV_Koppels.evaluate(kfolds, iter_l, featr_size_lst, sigma_threshold_list)
    
    CrossVal_Kopples_method_res.close()

    
    
    
    
    
    
    
    
    
    
    

