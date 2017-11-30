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
from dogswrapper.wrappedmodels import nnrd


# Santini's 7-genres Corpus
corpus_filepath = "/media/dimitrios/TurnstoneDisk/KI-04/"
state_saving_path = "/media/dimitrios/TurnstoneDisk/KI-04/" +\
    "Openness_C4G_KI04/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

# genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
genres = [
    "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(state_saving_path + 'Openness_OCSVME_C4G_Gensim_KI04_TESTONLY.h5', 'a')

params_range = coll.OrderedDict([
    ('vocab_size', [500]),
    ('features_size', [0]),
    ('split_ptg', [0.7]),
    ('ukwn_slt_ptg', [0.3]),
    ('rt_lims_stp', [[0.5, 1.0, 0.2]]),
    ('lmda', [0.2, 0.5, 0.7]),
    ('dims', [50, 100, 250]),
    ('min_trm_fq', [3]),
    ('win_size', [8]),
    ('algo', ['PV-DBOW']),
    ('alpha', [0.025]),
    ('min_alpha', [0.025]),
    ('epochs', [3]),
    ('decay', [0.002]),
    ('uknw_ctgs_num', [1, 4]),
    ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])

# word_n_gram_size = 1
# tables_wng = h2v_wng.Html2TF(
#     word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

char_n_gram_size = 4

# tables_cng = h2v_cng.Html2TF(
tables_cng = h2v_cng.Html2GsmVec(
    char_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
)

openness_model = nnrd

openness_searchgrid = OpennessParamGridSearchTables(
    openness_model, tables_cng, params_range, genres, corpus_filepath, method_results,
    state_saving_path, error_handling='replace', encoding='utf-8',
    # norm_func=MaxNormalise,
    norm_func=None,
)

openness_searchgrid.create_openness_iset()

# openness_searchgrid.build_vocabulary_on_openness_iset()
# openness_searchgrid.build_corpusmatrix_on_openness_iset()
# results_h5 = openness_searchgrid.evaluate_on_openness_iset()

openness_searchgrid.build_corpusmatrix_on_dlparams()
results_h5 = openness_searchgrid.evaluate_on_openness_dlparams()

# openness_searchgrid.build_corpusmatrix_on_dlparams_per_gnr()
# results_h5 = openness_searchgrid.evaluate_on_openness_dlparams_per_gnr()

print results_h5
print "OpenNNDR Experiments Done!"


method_results.close()
