""" """

import sys
sys.path.append('../../synergeticprocessing/src')
from generatevectors import Html2TF_Concurrent, Text2TF_Concurrent
#from synergeticpool import SynergeticPool

#pool = SynergeticPool( { '192.168.1.65':(40000,'123456'), '192.168.1.68':(40000,'123456') }, local_workers=1, syn_listener_port=41000 ) 
#pool = SynergeticPool( local_workers=1, syn_listener_port=41000 )
#print "Registering"
#pool.register_mod( ['html2vector', 'vectorhandlingtools', 'generatevectors'] )  
#print "Regitered OK"

#genres = [ "forum", "news", "product_companies", "blogs" ] #academic , "news", "wiki_pages", "product_companies", "blogs", "forum"  
#genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
#genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop"] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/saved_pages", "../Synergy-Crawler/saved_pages"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus", "../Synergy-Crawler/Santini_corpus"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/KI-04", "../Synergy-Crawler/KI-04"]  

base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt/"]
genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
#genres = [ "blog_pgs", "news_pgs", "product_pgs", "forum_pgs", "wiki_pgs" ] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_3000/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Manually_Selected_Crawled_corpus_75/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_500/"]


html2tf = Html2TF_Concurrent( lowercase=True )
html2nf = Html2TF_Concurrent(n=3, lowercase=True)
txt2tf = Text2TF_Concurrent( lowercase=True )
txt2nf = Text2TF_Concurrent(n=3, lowercase=True)

#if filepath and not os.path.isdir((filepath + "corpus_dictionaries/")):
#    os.mkdir((filepath + "corpus_dictionaries/"))
          
resaults = list()

"""
for g in genres:
    #Training Vectors file paths
    train_filepath = str( "/" + g + "/html_2500_pages/")
    train_tfv_file = str( "/" + g + "/train_tf_vectors/" + g + ".tfvl" )
    train_tfd_file = str( "/" + g + "/train_tf_dictionaries/" + g + ".tfd" )
    train_err_file = str( "/" + g + "/" + g + ".train.lst.err")
    #Testing Vectors file paths
    test_filepath = str( "/" + g + "/test_only_html_2500_pages/")
    test_tfv_file = str( "/" + g + "/test_tf_vectors/" + g + ".tfvl" )
    test_tfd_file = str( "/" + g + "/test_tf_dictionaries/" + g + ".tfd" )
    test_err_file = str( "/" + g + "/" + g + ".test.lst.err")
    resaults.append( pool.dispatch(\
    html2tf.exec_for, base_filepath, train_filepath, train_tfv_file, train_tfd_file, train_err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='replace', low_mem='True' \
                     ) )
    resaults.append( pool.dispatch(\
    html2tf.exec_for, base_filepath, test_filepath, test_tfv_file, test_tfd_file, test_err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='replace', low_mem='True' \
                     ) )
    
for res in resaults:
    print res.value
"""

for g in genres:
    #Vectors file paths
    #filepath = str( "/" + g + "/html2ascii_perl_text/")
    filepath = str( "/" + g + "/html2text_debian_text/")
    #filepath = str( "/" + g + "/htmldetagger_console_ver_text/")
    #filepath = str( "/" + g + "/htmldetagger_console_500_ver_text/")
    #filepath = str( "/" + g + "/txt_rapidminer_app/")
    #filepath = str( "/" + g + "/txt_Htmlremover_app/")
    #filepath = str( "/" + g + "/txt_html2vectors_mod/")
    #filepath = str( "/" + g + "/txt_html2vectors_mod_500/")
    #filepath = str( "/" + g + "/nltk-clean_html_text/")
    #filepath = str( "/" + g + "/lxml_elementtree_text/")
    #filepath = str( "/" + g + "/lxml_elementtree_text_500/")
    #filepath = str( "/" + g + "/html/")
    #tfv_file = str( "/" + g + "/html2ascii_perl_ng-tfv/" + g + ".nfvl" )
    tfv_file = str( "/" + g + "/html2text_debian_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/htmldetagger_console_ver_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/htmldetagger_console_ver_500_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/txt_rapidminer_app_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/txt_Htmlremover_app_ng-tfv/" + g + ".nfvl" ) 
    #tfv_file = str( "/" + g + "/txt_html2vectors_mod_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/txt_html2vectors_mod_500_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/nltk-clean_html_text_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/lxml_elementtree_text_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/lxml_elementtree_text_500_ng-tfv/" + g + ".nfvl" )
    #tfv_file = str( "/" + g + "/html_ng-tfv/" + g + ".nfvl" )
    tfd_file = str( "/" + g + "/nf_dictionaries/" + g + ".nfd" )
    err_file = str( "/" + g + "/" + g + ".train.lst.err")
    #Training Vectors file paths
    #train_filepath = str( "/" + g + "/html_2500_pages/")
    train_filepath = str( "/" + g + "/html_pages/")
    train_tfv_file = str( "/" + g + "/train_nf_vectors/" + g + ".nfvl" )
    train_tfd_file = str( "/" + g + "/train_nf_dictionaries/" + g + ".nfd" )
    train_err_file = str( "/" + g + "/" + g + ".train.lst.err")
    #Testing Vectors file paths
    #test_filepath = str( "/" + g + "/test_only_html_2500_pages/")
    test_filepath = str( "/" + g + "/test_only_html_pages/")
    test_tfv_file = str( "/" + g + "/test_nf_vectors/" + g + ".nfvl" )
    test_tfd_file = str( "/" + g + "/test_nf_dictionaries/" + g + ".nfd" )
    test_err_file = str( "/" + g + "/" + g + ".test.lst.err")
    #resaults.append( pool.dispatch(\
    #print html2nf.exec_for(base_filepath, train_filepath, train_tfv_file, train_tfd_file, train_err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='replace', low_mem='True')
    print txt2nf.exec_for(base_filepath, filepath, tfv_file, tfd_file, err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='replace', low_mem='True')
    #                 ) )
    #resaults.append( pool.dispatch(\
    #print html2nf.exec_for(base_filepath, test_filepath, test_tfv_file, test_tfd_file, test_err_file, load_encoding='utf-8', save_encoding='utf-8', error_handling='replace', low_mem='True')
    #                 ) )
       
#for res in resaults:
#    print res.value


#pool.join_all()

print "Thank you and Goodbye!"


