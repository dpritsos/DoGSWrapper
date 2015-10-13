

import sys
#sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
sys.path.append('../src')

import collections as coll

import tables as tb

#import html2vect.sparse.cngrams as h2v_cng
#import html2vect.sparse.wngrams as h2v_wcng
import html2vect.tables.cngrams as h2v_cng
#import html2vect.tables.wngrams as h2v_wcng

from base.paramgridcrossval import ParamGridCrossValBase, ParamGridCrossValTables
from wrappedmodels.ocsvme import OCSVME_Wrapped
    

#"Santini's 7-genres Corpus"
corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
kfolds_vocs_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Kfolds_Vocs_Inds_4Chars_7Genres"
genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage" ]
test_only_tgs = False
method_results = tb.open_file('/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/OCSVM_4Chars_7Genres.h5', 'w')

params_range = coll.OrderedDict([
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),\
    ('vocab_size', [5000, 10000, 50000, 100000]),\
    ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),\
    ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9])
])

#word_n_gram_size = 1
#tables_wng = h2v_wcng.Html2TF(word_n_gram_size, html_attrib='text', lowercase=True, valid_html=False)

char_n_gram_size = 4
tables_cng = h2v_cng.Html2TF(char_n_gram_size, html_attrib='text', lowercase=True, valid_html=False)

#For Tables
ml_model = OCSVME_Wrapped(genres)

pgrid_corssv = ParamGridCrossValTables(\
                    ml_model, tables_cng, method_results, 
                    genres, corpus_filepath, kfolds_vocs_filepath\
               )
               
html_file_l, cls_gnr_tgs = pgrid_corssv.corpus_files_and_tags()

results_h5 = pgrid_corssv.evaluate(html_file_l, cls_gnr_tgs, test_only_tgs, None, params_range, 'utf-8')

print results_h5

method_results.close()
