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
from dogswrapper.evalmethods.openset import OpenSetParamGridSearchTables
import dogswrapper.evalmethods.openness as openness
from dogswrapper.tools.normalisers import MaxNormalise, SubSamplingNorm
from dogswrapper.wrappedmodels import ocsvme, rfse

# Santini's 7-genres Corpus
corpus_filepath = "/mnt/turnstone/KI-04/"
state_saving_path = "/mnt/turnstone/KI-04/" +\
    "Openness_C4G_KI04/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

# genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
genres = [
    "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(
    state_saving_path + 'OpenSet_MarkedUknown_RFSE_C4G_Gensim_KI04_2018_07_23.h5', 'a'
)

params_range = coll.OrderedDict([
    ('terms_type', ['C4G']),
    ('vocab_size', ['NA']),
    ('dims', [50, 100, 250, 500, 1000]),
    ('min_trm_fq', [3, 10]),
    ('win_size', [3, 8, 20]),
    ('algo', ['PV-DBOW']),
    ('alpha', [0.025]),
    ('min_alpha', [0.025]),
    ('epochs', [1, 3, 10]),
    ('decay', [0.002, 0.02]),
    # ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
    # ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    ('features_size', [25, 50, 100]),
    ('sim_func', ['cosine_sim']),
    ('Sigma', [0.5, 0.7, 0.9]),
    ('Iterations', [10, 50, 100]),
    # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
    ('marked_uknw_ctg_lst', [1]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])

n_gram_size = 1
#tables_wng = h2v_wng.Html2TF(
tables_cng = h2v_cng.Html2GsmVec(
    n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
)

openset_model = rfse
# openness_model = OCSVMEDMPG_Wrapped(genres)

openset_unoise_searchgrid = OpenSetParamGridSearchTables(
    openset_model, tables_cng, params_range, genres, corpus_filepath, method_results,
    state_saving_path, error_handling='replace', encoding='utf-8',
    # norm_func=MaxNormalise,
    norm_func=None,
)

openset_unoise_searchgrid.create_open_unstrd_noise_iset()

# openset_unoise_searchgrid.build_vocabulary_on_open_unstrd_noise_iset()
# openset_unoise_searchgrid.build_corpusmatrix_on_open_unstrd_noise_iset()

# FOR BUILDING THE VOCABS ##############
"""
openness_searchgrid = openness.OpennessParamGridSearchTables(
    openness_model, tables_wng, params_range, genres, corpus_filepath, method_results,
    state_saving_path, error_handling='replace', encoding='utf-8',
    # norm_func=MaxNormalise,
    norm_func=None,
)
openness_searchgrid.build_corpusmatrix_on_dlparams()
"""
# FOR BUILDING THE VOCABS ##############

results_h5 = openset_unoise_searchgrid.evaluate_on_open_unstrd_noise_iset()

print results_h5
print "OCSVME Experiments Done!"


method_results.close()
