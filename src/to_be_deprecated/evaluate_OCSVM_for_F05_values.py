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
                                        
    def evaluate(self, kfolds, nu_list, featr_size_lst):
            
        for featr_size in featr_size_lst:
             
            for nu in nu_list:
                
                for k in range(kfolds):
                    
                    #Get the predicted classes per iteration and the expected classes 
                    predicted_Y_per_gnr = self.h5_res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='predicted_Y_per_gnr' )
                    predicted_Y_per_gnr = predicted_Y_per_gnr.read()
                    expected_Y = self.h5_res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu), name='expected_Y' )
                    expected_Y = expected_Y.read()
                    
                    #Store the Results for this Nu
                    print np.histogram(expected_Y, bins=np.arange(self.gnrs_num+2))
                    cv_tg_idxs = np.array( np.histogram(expected_Y, bins=np.arange(self.gnrs_num+2))[0], dtype=np.float)
                        
                    P_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                    R_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                    F05_per_gnr = np.zeros(self.gnrs_num+1, dtype=np.float)
                    
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
                    
                        #Calculate Recall
                        R_per_gnr[gnr_cnt+1] = tp_cnts_per_gnr / cv_tg_idxs[gnr_cnt+1]
                
                        #Calculate F1 score 
                        F05_per_gnr[gnr_cnt+1] = 1.25 * P_per_gnr[gnr_cnt+1] * R_per_gnr[gnr_cnt+1] / ((0.25 * P_per_gnr[gnr_cnt+1]) + R_per_gnr[gnr_cnt+1]) 
                         
                    F05_per_gnr[0] = 0  
                    
                    #Maybe Later
                    #fpr, tpr, thresholds = roc_curve(expected_Y, predicted_Y)   
                    
                    #Creating a Group for this Sigma
                    storage_grp = self.h5_res.getNode('/KFold'+str(k)+'/Feat'+str(featr_size)+'/Nu'+str(nu))
                    
                    print self.h5_res.createArray(storage_grp, "F05_per_gnr", F05_per_gnr, "F05_statistic per Genre (F05[0]==0)")[:]
                    print                
                                 
                                        
               
if __name__ == '__main__':
    
    corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
    #corpus_filepath = "/home/dimitrios/Synergy-Crawler/KI-04/"
    genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
    #genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop" ]
    CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/C-Santinis_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'a')
    #CrossVal_Kopples_method_res = tb.openFile('/home/dimitrios/Synergy-Crawler/KI-04/C-KI-04_TT-Words-OC-SVM_kfolds-10_Nu-Var_TM-TF.h5', 'a')
    
    kfolds = 10
    featr_size_lst = [1000, 5000, 10000, 20000, 50000, 70000] 
    nu_l = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
    
    eval_crossV_Koppels = Eval_CrossVal_Koppels_method(CrossVal_Kopples_method_res, corpus_filepath, genres)
    
    eval_crossV_Koppels.evaluate(kfolds, nu_l, featr_size_lst)
    
    CrossVal_Kopples_method_res.close()

    
    
    
    
    
    
    
    
    
    
    

