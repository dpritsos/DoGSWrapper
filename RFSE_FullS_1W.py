

import sys
#sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors/src')
sys.path.append('../src')

import collections as coll

import tables as tb

#import html2vect.sparse.cngrams as h2v_cng
#import html2vect.sparse.wngrams as h2v_wcng
#import html2vect.tables.cngrams as h2v_cng
import html2vect.tables.wngrams as h2v_wcng

from base.paramgridcrossval import ParamGridCrossValBase, ParamGridCrossValTables
from wrappedmodels.rfse import RFSE_Wrapped, cosine_similarity, cosine_similarity_sparse


#SANTINIS
corpus_filepath = "/home/dimitrios/Synergy-Crawler/SANTINIS/"
kfolds_vocs_filepath = "/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_1Words_SANTINI"
genres = [ "blog", "eshop", "faq", "frontpage", "listing", "php", "spage", "diy_mini", "editorial", "feat_articles", "short_bio", "spirit_1000" ]
#genres = [ "short_bio", "spirit_1000" ]
test_only_tgs = [12]
method_results = tb.open_file('/home/dimitrios/Synergy-Crawler/SANTINIS/RFSE_1Words_SANTINIS_TEST_TEST.h5', 'w')


params_range = coll.OrderedDict([
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),\
    ('vocab_size', [5000, 10000, 50000, 100000]),\
    ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),\
    ('Sigma', [0.5, 0.7, 0.9]),\
    ('Iterations', [10, 50, 100]),\
    #('Bagging', [0.66]),\
])

word_n_gram_size = 1
tables_wng = h2v_wcng.Html2TF(word_n_gram_size, html_attrib='text', lowercase=True, valid_html=False)

#char_n_gram_size = 4
#tables_cng = h2v_cng.Html2TF(char_n_gram_size, html_attrib='text', lowercase=True, valid_html=False)

ml_model = RFSE_Wrapped(cosine_similarity, -1.0, genres[0:-1], bagging=False)

pgrid_corssv = ParamGridCrossValTables(\
                    ml_model, tables_wng, method_results,
                    genres, corpus_filepath, kfolds_vocs_filepath\
               )

html_file_l, cls_gnr_tgs = pgrid_corssv.corpus_files_and_tags()

results_h5 = pgrid_corssv.evaluate(html_file_l, cls_gnr_tgs, test_only_tgs, None, params_range, 'utf-8')

print results_h5

method_results.close()
