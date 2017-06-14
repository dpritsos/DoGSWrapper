# -*- coding: utf-8 -*-

import collections as coll
import tables as tb
import os
import sys

# sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors')
sys.path.append('../dogswrapper')

# import html2vect.sparse.cngrams as h2v_cng
# import html2vect.sparse.wngrams as h2v_wng
# import html2vect.tables.cngrams as h2v_cng
import html2vect.tables.wngrams as h2v_wng
# NOTE: base.opensetcrossval_py
from base.opensetcrossval import OpenSetParamGridSearchBase, OpenSetParamGridSearchTables
from wrappedmodels.nnrd import nnrd_eval
from wrappedmodels.rfse import minmax_similarity


# Santini's 7-genres Corpus
corpus_filepath = "/media/dimitrios/TurnstoneDisk/KI-04/"
state_saving_path = "/media/dimitrios/TurnstoneDisk/KI-04/" +\
    "Openness_NNRD_W1G_KI04/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

# genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
genres = [
    "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(state_saving_path + 'Openness_NNRD_W1G_KI04_2017_TESTING_DROP_IT.h5', 'a')

params_range = coll.OrderedDict([
    ('vocab_size', [100000]),
    # ('features_size', [10000]),
    ('split_ptg', [0.5]),
    ('ukwn_slt_ptg', [0.5]),
    ('rt_lims_stp', [[0.3, 0.5, 0.7]]),
    ('lmda', [0.5]),
    ('uknw_ctgs_num', [1]),
    ('uknw_ctgs_num_splt_itrs', [0]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])

# word_n_gram_size = 1
# # RootTerm_size = 4
# tables_wng = h2v_wng.Html2TF(
#     word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

word_n_gram_size = 1
tables_wng = h2v_wng.Html2TF(
    word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
)

# char_n_gram_size = 4
# tables_cng = h2v_cng.Html2TF(
#     char_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

openness_searchgrid = OpenSetParamGridSearchTables(
    nnrd_eval, tables_wng, genres, method_results, corpus_filepath, state_saving_path
)

results_h5 = openness_searchgrid.EvaluateAll(params_range, encoding='utf-8')

print results_h5

method_results.close()
