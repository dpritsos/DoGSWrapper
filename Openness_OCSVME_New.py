# -*- coding: utf-8 -*-

import collections as coll
import tables as tb
import os
import sys

# sys.path.append('../../synergeticprocessing/src')
sys.path.append('../')
# import html2vec.sparse.cngrams as h2v_cng
# import html2vec.sparse.wngrams as h2v_wcng
import html2vec.tables.cngrams as h2v_cng
# import html2vec.tables.wngrams as h2v_wng
from dogswrapper.evalmethods.openness import OpennessParamGridSearchTables
from dogswrapper.tools.normalisers import MaxNormalise, SubSamplingNorm
from dogswrapper.wrappedmodels.ocsvme import OCSVME_Wrapped


# Santini's 7-genres Corpus
corpus_filepath = "/media/dimitrios/TurnstoneDisk/KI-04/"
state_saving_path = "/media/dimitrios/TurnstoneDisk/KI-04/" +\
    "Openness_OCSVME_C4G_KI04_8Iter_MaxNorm_TESTINGCODE_2017_10_11/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

# genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
genres = [
    "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(state_saving_path + 'Openness_OCSVME_C4G_KI04_1Iter_TESTINGCODE_2017_10_11.h5', 'a')

params_range = coll.OrderedDict([
    ('vocab_size', [500]),
    ('features_size', [100]),
    ('nu', [0.07]),  # 0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9
    ('uknw_ctgs_num', [1]),
    ('uknw_ctgs_num_splt_itrs', [0, 1]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])

# word_n_gram_size = 1
# tables_wng = h2v_wng.Html2TF(
#     word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

char_n_gram_size = 4
tables_cng = h2v_cng.Html2TF(
    char_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
)

openness_model = OCSVME_Wrapped(genres)

openness_searchgrid = OpennessParamGridSearchTables(
    openness_model, tables_cng, params_range, genres, corpus_filepath, method_results,
    state_saving_path, norm_func=MaxNormalise, error_handling='replace', encoding='utf-8'
)

# openness_searchgrid.create_openness_iset()
# openness_searchgrid.build_vocabulary_on_openness_iset()
# openness_searchgrid.build_corpusmatrix_on_openness_iset()
results_h5 = openness_searchgrid.evaluate_on_openness_iset()

print results_h5

method_results.close()
