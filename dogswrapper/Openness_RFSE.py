# -*- coding: utf-8 -*-

import collections as coll
import tables as tb
import os
import sys

# sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors')
sys.path.append('../dogswrapper')

# import html2vect.sparse.cngrams as h2v_cng
# import html2vect.sparse.wngrams as h2v_wcng
import html2vect.tables.cngrams as h2v_cng
# import html2vect.tables.wngrams as h2v_wcng
from base.opensetcrossval import OpenSetParamGridSearchBase, OpenSetParamGridSearchTables
from wrappedmodels.rfse import RFSE_Wrapped, cosine_similarity, cosine_similarity_sparse
from wrappedmodels.rfse import minmax_similarity


# Santini's 7-genres Corpus
corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
state_saving_path = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/Openess_4Chars_7Genres"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(state_saving_path + 'Openness_RFSE_C4G_7Genres_Cos.h5', 'a')

params_range = coll.OrderedDict([
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ('onlytest_gnrs_splts', [1, 2, 3, 4]),
    ('onlytest_splt_itrs', [0, 1, 2]),
    ('vocab_size', [10000]),
    ('features_size', [1000]),
    ('Sigma', [0.5]),
    ('Iterations', [100])
])

# word_n_gram_size = 3
# tables_wng = h2v_wcng.Html2TF(
#    word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

char_n_gram_size = 4
tables_cng = h2v_cng.Html2TF(
    char_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
)

openness_model = RFSE_Wrapped(cosine_similarity, -1.0, genres, bagging=False)
# If no test_only_tgs is given as input 'genres' tag list should be given full not genres[0:-1]

openness_searchgrid = OpenSetParamGridSearchTables(
    openness_model, tables_cng, genres, method_results, corpus_filepath, state_saving_path
)

results_h5 = openness_searchgrid.EvaluateAll(params_range, encoding='utf-8')

print results_h5

method_results.close()
