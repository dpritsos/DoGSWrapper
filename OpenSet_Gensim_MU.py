# -*- coding: utf-8 -*-

import collections as coll
import tables as tb
import os
import sys

# import html2vec.sparse.cngrams as h2v_cng
# import html2vec.sparse.wngrams as h2v_wcng
# import html2vec.tables.cngrams as h2v_cng
import html2vec.tables.wngrams as h2v_wng
from dogswrapper.evalmethods.openset import OpenSetParamGridSearchTables
import dogswrapper.evalmethods.openness as openness
from dogswrapper.tools.normalisers import MaxNormalise, SubSamplingNorm
from dogswrapper.wrappedmodels import ocsvme, rfse, nnrd, svmro


# Santini's 7-genres Corpus
corpus_filepath = "/home/dimitrios/Synergy-Crawler/SANTINIS/"
state_saving_path = "/home/dimitrios/Synergy-Crawler/SANTINIS/" +\
    "Open_W3G_SANTINIS/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

# genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
# genres = [
#   "article", "discussion", "download", "help", "linklist", "portrait",
#   "portrait_priv", "shop"
# ]

genres = [
    "blog", "eshop", "faq", "frontpage", "listing", "php", "spage", "diy_mini",
    "editorial", "feat_articles", "short_bio", "spirit_1000"
]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(
    state_saving_path + 'OpenSet_MarkedUknown_SVMRO_W3G_Gensim_SANTINIS_2018_08_30.h5', 'a'
)

params_range = coll.OrderedDict([
    ('terms_type', ['W3G']),
    ('vocab_size', ['NA']),
    ('dims', [50, 100, 250, 500, 1000]),
    ('min_trm_fq', [3, 10]),
    ('win_size', [3, 8, 20]),
    ('algo', ['PV-DBOW']),
    ('alpha', [0.025]),
    ('min_alpha', [0.025]),
    ('epochs', [1, 3, 10]),
    ('decay', [0.002, 0.02]),
    # openness
    # ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
    # ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    # RFSE
    # ('features_size', [25, 50, 100]),
    # ('sim_func', ['minmax_sim']),
    # ('Sigma', [0.5, 0.7, 0.9]),
    # ('Iterations', [10, 50, 100]),
    # OCSVME
    # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
    # NNRD
    # ('split_ptg', [0.7, 0.5]),
    # ('ukwn_slt_ptg', [0.3, 0.5]),
    # ('rt_lims_stp', [[0.8, 1.0, 0.2]]),
    # ('lmda', [0.2, 0.5, 0.7]),
    # SVMRO
    ('svm_type', ['oneclass']),
    # ('svm_type', ['binary']),
    ('ll', [0.3, 0.8]),
    ('c1_w', [0.3, 0.7]),
    ('c2_w', [0.3, 0.7]),
    ('mrgn_nw', [0.3, 0.7]),
    ('mrgn_fw', [0.3, 0.7]),
    # SVMRO oneclass
    ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
    ('marked_uknw_ctg_lst', [12]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])

word_n_gram_size = 3
tables_wng = h2v_wng.Html2GsmVecBin(
# tables_cng = h2v_cng.Html2GsmVecBin(
# tables_wng = h2v_wng.Html2GsmVec(
# tables_cng = h2v_cng.Html2GsmVec(
    word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
)
# openness_model = OCSVMEDMPG_Wrapped(genres)

openness_model = svmro

openset_unoise_searchgrid = OpenSetParamGridSearchTables(
    openness_model, tables_wng, params_range, genres, corpus_filepath, method_results,
    state_saving_path, error_handling='replace', encoding='utf-8',
    # norm_func=MaxNormalise,
    norm_func=None,
)

openset_unoise_searchgrid.create_open_unstrd_noise_iset()

# openset_unoise_searchgrid.build_vocabulary_on_open_unstrd_noise_iset()
# openset_unoise_searchgrid.build_corpusmatrix_on_open_unstrd_noise_iset()

# FOR BUILDING THE VOCABS ###############

openness_searchgrid = openness.OpennessParamGridSearchTables(
    openness_model, tables_wng, params_range, genres, corpus_filepath, method_results,
    state_saving_path, error_handling='replace', encoding='utf-8',
    # norm_func=MaxNormalise,
    norm_func=None,
)
openness_searchgrid.build_corpusmatrix_on_dlparams()

print "Building the Vocabs/CorpusMatricies: " + state_saving_path + " Done!"

# FOR BUILDING THE VOCABS ###############

"""
results_h5 = openset_unoise_searchgrid.evaluate_on_open_unstrd_noise_iset()

print results_h5
print "Experiments Done!"
"""


method_results.close()
