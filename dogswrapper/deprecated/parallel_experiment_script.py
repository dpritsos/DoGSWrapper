"""

"""
import os
import sys
sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
from vectorhandlingtools import *
import sys
from synergeticpool import *
from experiments import *

#pool = SynergeticPool( { '192.168.1.68':(40000,'123456') }, local_workers=1, syn_listener_port=41000 )
#pool = SynergeticPool( { '192.168.1.68':(40000,'123456'), '192.168.1.65':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
pool = SynergeticPool( local_workers=1, syn_listener_port=41000 )
print "Registering"
#pool.register_mod( [ 'basicfiletools', 'vectorhandlingtools', 'trainevaloneclssvm', 'experiments'] )  
print "Regitered OK"
exp = SVMExperiments()

#genres = [  "wiki_pages", "product_companies", "forum", "blogs", "news" ] #academic , "forum",   
#genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"] 
#genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop"] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/saved_pages/", "../Synergy-Crawler/saved_pages/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus/", "../Synergy-Crawler/Santini_corpus/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt/", "../Synergy-Crawler/Santini_corpus_html2txt/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/KI-04/", "../Synergy-Crawler/KI-04/"] 
train_tf_d = "/train_tf_dictionaries/"
train_tf_vectors = "/train_tf_vectors/"
test_tf_d = "/test_tf_dictionaries/"
test_tf_vectors = "/test_tf_vectors/"
##
train_nf_d = "/train_nf_dictionaries/"
train_nf_vectors = "/train_nf_vectors/"
test_nf_d = "/train_nf_dictionaries/"
test_nf_vectors = "/test_nf_vectors/"

#genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt/"]
genres = [ "blog_pgs", "news_pgs", "product_pgs", "forum_pgs", "wiki_pgs" ] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_3000/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Manually_Selected_Crawled_corpus_75/"]
base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_500/"]


## Paths of Vector Lists for Santini's Corpus Cleaned-up from HTML tags by 8 different tools 
train_nf_vectors = "/txt_html2vectors_mod_ng-tfv/"
#train_nf_vectors = "/txt_html2vectors_mod_500_ng-tfv/"
#train_nf_vectors = "/txt_Htmlremover_app_ng-tfv/"
#train_nf_vectors = "/lxml_elementtree_text_ng-tfv/"
#train_nf_vectors = "/lxml_elementtree_text_500_ng-tfv/"
#train_nf_vectors = "/nltk-clean_html_text_ng-tfv/"
#train_nf_vectors = "/html2ascii_perl_ng-tfv/"
#train_nf_vectors = "/htmldetagger_console_ver_ng-tfv/"
#train_nf_vectors = "/htmldetagger_console_ver_500_ng-tfv/"
#train_nf_vectors = "/txt_rapidminer_app_ng-tfv/"
#train_nf_vectors = "/html2text_debian_ng-tfv/"
#train_nf_vectors = "/html_ng-tfv/"

#test_nf_vectors = "/lxml_elementtree_text_ng-tfv/" #For Now!
#test_nf_vectors = "/test/"



exps_report = list()
#for g in genres:
#    exps_report.append( pool.dispatch(exp.tf_experiment_set1, base_filepath, train_tf_vectors, 140, test_tf_vectors, 49, g, genres, lower_case=True) )
#    print("Experiment %s VS All: Dispatched" % g)

#Simple OC-SVM with specific Experimental Variable Attributes
#nu_l = [0.07, 0.07, 0.07, 0.05, 0.05] #[0.05, 0.07, 0.05, 0.05, 0.05 ] 
#featr_l = [40, 160, 40, 70, 10] #[10, 10, 40, 10, 40 ]
#vect_format = 2
#for g, nu, featrs in zip(genres, nu_l, featr_l):
#    exps_report.append( pool.dispatch(exp.exprmt_ocsvm, nu, featrs, vect_format, base_filepath, train_nf_vectors, 2500, test_nf_vectors, 500, g, genres, lower_case=True) )
#    print("Experiment %s VS All: Dispatched" % g)

################################################
#keep_term_lst =  [20, 50, 80, 100, 150, 200, 300]#range(500, 63000, 1500)
#keep_term_lst.extend(range(500, 63000, 1500))
#keep_term_lst = range(10, 720, 50)
#for g in genres:
#    exps_report.append( pool.dispatch(exp.exprmt_feature_len_variation, keep_term_lst, base_filepath, train_nf_vectors, 66, test_nf_vectors, 9, g, genres, lower_case=True) )
#    print("Experiment %s VS All: Dispatched" % g)

#keep_term_lst = range(10, 670, 30)
#nu_lst = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8]
#kfolds_l = [37, 30, 19, 10, 41]
#resdualz_l = [21, 10, 30, 0, 40]
#for i, g in enumerate(genres):
#    exps_report.append( pool.dispatch(exp.kfold_corss_v_featr_nu, kfolds_l[i], keep_term_lst, base_filepath, train_tf_vectors, 2500 - resdualz_l[i], nu_lst, g, lower_case=True) )
#    print("Experiment %s VS All: Dispatched" % g)
    
#for g in genres:
#    exps_report.append( pool.dispatch(exp.tf_experiment_set4, base_filepath, train_tf_vectors, 2500, test_tf_vectors, 800, g, genres, freq_init=5, freq_lim=200, freq_step=10,lower_case=True, keep_terms=None) )
#    print("Experiment %s VS All: Dispatched" % g)


#########################
keep_term_lst =  [1000] #[10, 20, 50, 80, 100, 150, 200, 250, 300, 400]#range(500, 63000, 1500)
#keep_term_lst.extend(range(500, 63000, 1500))
#keep_term_lst = [3500] #range(1500, 47000, 3000)
c_lst = [1] #[1, 2, 5, 10, 50]
kfolds = 10
exps_report.append( pool.dispatch(exp.kfold_corss_multiclass, kfolds, keep_term_lst, c_lst, base_filepath, train_nf_vectors, 0, test_nf_vectors, 0, genres, lower_case=True) )
#exps_report.append( pool.dispatch(exp.Testing_mem_phase1, kfolds, keep_term_lst, c_lst, base_filepath, train_nf_vectors, 0, test_nf_vectors, 0, genres, lower_case=True) )
#exps_report.append( pool.dispatch(exp.Testing_mem_phase2, kfolds, keep_term_lst, c_lst, base_filepath, train_nf_vectors, 0, test_nf_vectors, 0, genres, lower_case=True) )
#exps_report.append( pool.dispatch(exp.Testing_mem_phase3, kfolds, keep_term_lst, c_lst, base_filepath, train_nf_vectors, 0, test_nf_vectors, 0, genres, lower_case=True) )
##########################


print("Experiment k-fold Cross-Validation Multi-Class SVM Dispatched")

#keep_term_lst = [500, 2500, 5000, 60000] #range(500, 63000, 1500)
#c_lst = [2, 10]
#exps_report.append( pool.dispatch(exp.svm_multiclass_featrs, keep_term_lst, base_filepath, train_nf_vectors, 1000, test_nf_vectors, 500, c_lst, genres, lower_case=True ) )
##exps_report.append( pool.dispatch(exp.tf_experiment_set2, base_filepath, train_tf_vectors, 1000, test_tf_vectors, 500, genres, lower_case=True) )
#print("Experiment Multi-Class SVM Dispatched")

for exp_rep in exps_report:
    print exp_rep.value    

pool.join_all()

print "Thank you and Goodbye!"

