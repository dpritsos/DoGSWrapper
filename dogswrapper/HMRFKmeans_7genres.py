
import collections as coll
import tables as tb
import sys
import os
# sys.path.append('../../synergeticprocessing/src')
sys.path.append('../../html2vectors')
sys.path.append('../dogswrapper')
from base.clusteringcrossval import SemiSupervisedParamGridSearchBase
from base.clusteringcrossval import SemiSupervisedParamGridSearchTables
from wrappedmodels.HMRFKmeans import HMRFKmeans_Wrapped
# import html2vect.sparse.cngrams as h2v_cng
import html2vect.sparse.wngrams as h2v_wcng
# import html2vect.tables.cngrams as h2v_cng
# import html2vect.tables.wngrams as h2v_wcng


# "Santini's 7-genres Corpus"
corpus_filepath = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/"
state_saving_path = "/home/dimitrios/Synergy-Crawler/Santinis_7-web_genre/SemiSupClust_1W_7Genres/"
if not os.path.exists(state_saving_path):
    os.mkdir(state_saving_path)

genres = ["blog", "eshop", "faq", "frontpage", "listing", "php", "spage"]
method_results = tb.open_file(state_saving_path + 'HMRFKmeans_1W_7Genres.h5', 'w')

params_range = coll.OrderedDict([
   #  ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
   ('train_split_step_method', [
      (0.5, 0.1, 'rndred_trn_rest4_test'),
      (0.5, 0.1, 'rndred_trn_fixed_test'),
   ]),
   ('vocab_size', [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]),
   ('max_iter', [50, 100, 300]),
   ('converg_diff', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
   ('learing_rate', [0.0003, 0.003, 0.01, 0.03, 0.1, 0.3])
])

word_n_gram_size = 1
tables_wng = h2v_wcng.Html2TF(
    word_n_gram_size, html_attrib='text', lowercase=True, valid_html=False
)

# char_n_gram_size = 4
# tables_cng = h2v_cng.Html2TF(
#     char_n_gram_size, html_attrib='text', lowercase=True, valid_html=False
# )

semisupervised_model = HMRFKmeans_Wrapped()

SemiSPGridSearch = SemiSupervisedParamGridSearchBase(
                    semisupervised_model, tables_wng, genres,
                    method_results, corpus_filepath, state_saving_path
               )

results_h5 = SemiSPGridSearch.EvaluateAll(params_range, encoding='utf-8')

print results_h5

results_h5.close()
