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
# import html2vect.tables.cngrams as h2v_cng
import html2vect.tables.wngrams as h2v_wng
from base.opensetcrossval_py import OpenSetParamGridSearchBase, OpenSetParamGridSearchTables

from wrappedmodels.ovssvm import LOPSVM_Wrapped


# Santini's 7-genres Corpus
corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
state_saving_path = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/" +\
    "LOPSVM_BIN_Openness_W1G_RTV3_7Genres/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
# genres = [
#    "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
# ]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(state_saving_path + 'LOPSVM_BIN_Openness_W1G_RTV3_7Genres.h5', 'a')

# Oneclass SVM
# params_range = coll.OrderedDict([
#     ('svm_type', ['oneclass']),
#     ('vocab_size', [100000]),
#     ('features_size', [1000, 5000, 10000]),
#     ('nu', [0.08, 0.1, 0.18, 0.3, 0.5, 0.7, 0.9]),
#     ('l', [0.3, 0.8]),
#     ('c1_w', [0.3, 0.7]),
#     ('c2_w', [0.3, 0.7]),
#     ('mrgn_nw', [0.3, 0.7]),
#     ('mrgn_fw', [0.3, 0.7]),
#     ('onlytest_gnrs_splts', [1, 2, 3, 4, 5, 6]),
#     ('onlytest_splt_itrs', [0, 1]),
#     ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
# ])

# Binary SVM
params_range = coll.OrderedDict([
    ('svm_type', ['binary']),
    ('vocab_size', [1000]),
    ('features_size', [1000]),
    ('mrgn_nw', [0.5]),
    ('mrgn_fw', [0.5]),
    ('onlytest_gnrs_splts', [1]),
    ('onlytest_splt_itrs', [0, 1, 2, 3]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])


word_n_gram_size = 1
RootTerm_size = 5
tables_wng = h2v_wng.Html2TV(
    word_n_gram_size, RootTerm_size, html_attrib=["text"], str_case='lower', valid_html=False
)

# word_n_gram_size = 1
# tables_wng = h2v_wcng.Html2TF(
#     word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

# char_n_gram_size = 4
# tables_cng = h2v_cng.Html2TF(
#     char_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

openness_model = LOPSVM_Wrapped(state_saving_path, genres)

openness_searchgrid = OpenSetParamGridSearchTables(
    openness_model, tables_wng, genres, method_results, corpus_filepath, state_saving_path
)

results_h5 = openness_searchgrid.EvaluateAll(params_range, encoding='utf-8')

print results_h5

method_results.close()
