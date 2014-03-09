""" """

import sys
sys.path.append('../../html2vectors/src')

import tables as tb
import numpy as np
from html2vect.base.convert import tfttools as tbtools
import html2vect.tables.cngrams as cngrams
import html2vect.tables.words as words


words.Html2TF(lowercase=False, valid_html=True, ndtype=np.dtype([('terms', 'S128'), ('freq', 'float32')]))

genres = [ "blogs", "forum", "news", "product_pages", "wiki_pages" ] #academic , "news", "wiki_pages", "product_companies", "blogs", "forum"  
#genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
#genres = [ "article", "discussion", "download", "help", "linklist", "portrait", "shop"] 
base_filepath = ["/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus", "../Synergy-Crawler/Automated_Crawled_Corpus"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre", "../Synergy-Crawler/Santinis_7-web_genre"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/KI-04", "../Synergy-Crawler/KI-04"]  

##base_filepath = ["/home/dimitrios/Synergy-Crawler/Santini_corpus_html2txt/"]
##genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
#genres = [ "blog_pgs", "news_pgs", "product_pgs", "forum_pgs", "wiki_pgs" ] 
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_3000/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Manually_Selected_Crawled_corpus_75/"]
#base_filepath = ["/home/dimitrios/Synergy-Crawler/Crawled_corpus_500/"]


html2tf_wrd = words.Html2TF(lowercase=True, valid_html=False, ndtype=np.dtype([('terms', 'S128'), ('freq', 'float32')]))
html2tf_3n = cngrams.Html2TF(n=3, lowercase=True, valid_html=False, ndtype=np.dtype([('terms', 'S3'), ('freq', 'float32')]))

#Creating the Default H5File tree structure to save the Corpus' TF and other Tables(pytables)
#Create HD5 file in user defined path
h5file = tb.openFile("/home/dimitrios/Synergy-Crawler/Automated_Crawled_Corpus/ACC.h5", mode="w")
#h5file = tb.openFile("/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Santini_corpus_w.h5", mode="w")
CorpusTable = tbtools.TFTablesTools(h5file)   
#CorpusTable.create(ttypes_structures_lst=["trigrams"], corpus_name="Automated_Crawled_Corpus", genres_lst=genres)
CorpusTable.create(ttypes_structures_lst=["trigrams"], corpus_name="Automated_Crawled_Corpus", genres_lst=genres)

print CorpusTable.get()

for g in genres:
    #Vectors file paths
    #filepath = str( "/" + g + "/html2ascii_perl_text/")
    #filepath = str( "/" + g + "/html2text_debian_text/")
    #filepath = str( "/" + g + "/htmldetagger_console_ver_text/")
    #filepath = str( "/" + g + "/htmldetagger_console_500_ver_text/")
    #filepath = str( "/" + g + "/txt_rapidminer_app/")
    #filepath = str( "/" + g + "/txt_Htmlremover_app/")
    #filepath = str( "/" + g + "/txt_html2vectors_mod/")
    #filepath = str( "/" + g + "/txt_html2vectors_mod_500/")
    #filepath = str( "/" + g + "/nltk-clean_html_text/")
    #filepath = str( "/" + g + "/lxml_elementtree_text/")
    #filepath = str( "/" + g + "/lxml_elementtree_text_500/")
    filepath = str( "/" + g + "/html/")
    #tfv_file = str( "/" + g + "/html2ascii_perl_ng-tfv/" + g + ".nfvl" )
    print base_filepath[0]+filepath
    #tablesGroup, GenrePageListTable = html2tf_wrd.from_paths() but I don't want what it returns right now 
    print html2tf_3n.from_paths(CorpusTable.get(), "/Automated_Crawled_Corpus/trigrams/"+g, "PageListTable",\
                                 None, base_filepath[0]+filepath, encoding='utf-8', error_handling='replace')
    
    #print html2tf_3n(base_filepath, filepath, CorpusTable.get(), "/Automated_Crawled_Corpus/trigrams/"+g, "PageListTable", load_encoding='utf-8', error_handling='replace')

       
#for res in resaults:
#    print res.value

CorpusTable.get().close()


#pool.join_all()

print "Thank you and Goodbye!"


