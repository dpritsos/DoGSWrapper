# -*- coding: utf-8 -*-

import collections as coll
import tables as tb
import os
import sys

# sys.path.append('../../synergeticprocessing/src')
sys.path.append('../')
# import html2vec.tables.cngrams as h2v_cng
# import html2vec.tables.wngrams as h2v_wng
import html2vec.tables.posngrams as h2v_pos
from dogswrapper.evalmethods.openset import OpenSetParamGridSearchTables
from dogswrapper.tools.normalisers import MaxNormalise, SubSamplingNorm
from dogswrapper.wrappedmodels import rfse, ocsvme, ecce
# from dogswrapper.wrappedmodels.rfse_py import RFSE_Wrapped, RFSEDMPG_Wrapped, cosine_similarity


# Santini's 7-genres Corpus
# corpus_filepath = "/media/dimitrios/TurnstoneDisk/SANTINIS/"
# state_saving_path = "/media/dimitrios/TurnstoneDisk/SANTINIS/" + "C4G_SANTINIS/"
corpus_filepath = "/home/dimitrios/Synergy-Crawler/SANTINIS/"
state_saving_path = "/home/dimitrios/Synergy-Crawler/SANTINIS/" + "POS_SANTINIS/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

# genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
# genres = [
#    "article", "discussion", "download", "help", "linklist", "portrait", "portrait_priv", "shop"
# ]

genres = [
    "blog", "eshop", "faq", "frontpage", "listing", "php", "spage", "diy_mini", "editorial",
    "feat_articles", "short_bio", "spirit_1000"
]

# Creating or opeding existing file for saving the results.
method_results = tb.open_file(
    state_saving_path + "ECCE_POS2G_SANTINIS_2018_03_15.h5", 'a'
)

params_range = coll.OrderedDict([
    ('terms_type', ['POS1G']),
    ('vocab_size', [43]),
    # 1330, 16200   10000, 50000, 100000, 5000, 10000, 50000,
    # ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),  # , 5000, 10000, 50000, 90000
    # 4, 10, 20, 40, 100, 500, 1000, 5000, 10000, 15000
    ('sim_func', ['cosine_sim', 'minmax_sim']), # 'minmax_sim'
    # ('Sigma', [0.5, 0.7, 0.9]),
    # ('Iterations', [10, 50, 100, 200, 300, 500, 1000]),  # 10, 50, 100, 200, 300, 500
    # ('nu', [0.05, 0.07, 0.1, 0.15, 0.17, 0.3, 0.5, 0.7, 0.9]),
    # ('dims', [50, 100, 250, 500, 1000]),
    # ('min_trm_fq', [3, 10]),
    # ('win_size', [3, 8, 20]),
    # ('algo', ['PV-DBOW']),
    # ('alpha', [0.025]),
    # ('min_alpha', [0.025]),
    # ('epochs', [1, 3, 10]),
    # ('decay', [0.002, 0.02]),
    # ('uknw_ctgs_num', [1, 2, 3, 4, 5, 6, 7]),
    # ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    ('marked_uknw_ctg_lst', [12]),
    ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
])

pos_n_gram_size = 3
tables_pos = h2v_pos.Html2TF(
    tagger_cls='english-bidirectional-distsim.tagger', n=pos_n_gram_size,
    html_attrib=["text"], str_case='lower', valid_html=False
)

# word_n_gram_size = 3
# tables_wng = h2v_wng.Html2GsmVec(
# tables_wng = h2v_wng.Html2TF(
#    word_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
#)

# char_n_gram_size = 4
# tables_cng = h2v_cng.Html2GsmVec(
# tables_cng = h2v_cng.Html2TF(
#    char_n_gram_size, html_attrib=["text"], str_case='lower', valid_html=False
# )

openset_model = ecce

openset_unoise_searchgrid = OpenSetParamGridSearchTables(
    openset_model, tables_pos, params_range, genres, corpus_filepath, method_results,
    state_saving_path, error_handling='replace', encoding='utf-8',
    norm_func=MaxNormalise,
    # norm_func=None,
)

openset_unoise_searchgrid.create_open_unstrd_noise_iset()

# openset_unoise_searchgrid.build_vocabulary_on_open_unstrd_noise_iset()
# openset_unoise_searchgrid.build_corpusmatrix_on_open_unstrd_noise_iset()

results_h5 = openset_unoise_searchgrid.evaluate_on_open_unstrd_noise_iset()

print results_h5

print "RFSE TF Openness Experiments Done!"

method_results.close()
