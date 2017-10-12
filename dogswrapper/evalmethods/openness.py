# -*- coding: utf-8 -*-

import json
import os
import cPickle as pickle
import numpy as np
import tables as tb
import scipy.sparse as ssp
import collections as coll

from ..tools.loaders import LoadCrpsFnamesTags
from ..tools.samplers import OpennessSplitSamples, SelectStratifiedKfolds
from ..tools.samplers import LoadSplitSamples, SaveSplitSamples
from ..tools.paramcombs import ParamGridIter

import sys
sys.path.append('../../../')
from html2vec.utils import tfdutils


class OpennessParamGridSearchTables(object):

    def __init__(
        self, model, terms_model, params_range, class_names_lst, corpus_fpath, h5f_res, state_path,
            norm_func, error_handling, encoding):

        self.model = model
        self.terms_model = terms_model
        self.params_range = params_range
        self.classes_lst = class_names_lst
        self.classes_num = len(class_names_lst)
        self.h5_res = h5f_res
        self.error_handling = error_handling
        self.encoding = encoding
        self.norm_func = norm_func

        self.corps_fpath = corpus_fpath
        if not os.path.exists(self.corps_fpath):
            raise Exception("Corpus files path does not exist.")

        self.state_path = state_path
        if not os.path.exists(self.state_path):
            print "Creating a new process-state save path..."
            os.mkdir(self.state_path)
            print "New process-state saving path is: '" + self.state_path + "'"

        # Loading the Filename list of the corpus and their respective class tags.
        self.html_file_l, self.cls_tgs = LoadCrpsFnamesTags(
            self.corps_fpath, self.state_path, self.classes_lst
        )

    def create_openness_iset(self):

        if not os.path.exists(self.corps_fpath):
            raise Exception("Corpus files path does not exist.")

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])
        kfolds = len(self.params_range['kfolds'])

        # Creating the Openness Kfold-Splits.
        # NOTE: KFolds here are given as a count not as part of Param Grid Iteratable sets.
        for params in ParamGridIter(coll.OrderedDict(self.params_range.items()[0:-2])):

            # Building the splits.
            train_splts, test_splts, onlyt_splts = OpennessSplitSamples(
                    self.cls_tgs,
                    onlytest_clsnum=params['uknw_ctgs_num'],
                    uknw_ctgs_num_splt_itrs=ukn_iters,
                    kfolds=kfolds
            )

            SaveSplitSamples(
                train_splts, test_splts, onlyt_splts,
                params['uknw_ctgs_num'], ukn_iters, self.state_path
            )

    def build_vocabulary_on_openness_iset(self):

        # Building the Vocabularies for all Splits/Folds
        print "Building Vocabularies..."

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])

        for params in ParamGridIter(self.params_range):

            train_splts, test_splts, onlyt_splts = LoadSplitSamples(
                params['uknw_ctgs_num'], ukn_iters, self.state_path
            )

            # Building the TF Vocabulary.
            tf_vocab = self.terms_model.build_vocabulary(
                list(
                    self.html_file_l[
                        train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
                    ]
                ),
                encoding=self.encoding, error_handling=self.error_handling
            )

            # Saving TF Vocabulary in pickle and Json format.
            split_suffix = '_S' + str(params['uknw_ctgs_num']) +\
                '_I' + str(params['uknw_ctgs_num_splt_itrs']) +\
                '_kF' + str(params['kfolds'])
            vocab_fname = self.state_path + 'Vocab' + split_suffix

            with open(vocab_fname + '.pkl', 'w') as f:
                pickle.dump(tf_vocab, f)

            with open(vocab_fname + '.jsn', 'w') as f:
                json.dump(tf_vocab, f, encoding=self.encoding)

    def build_corpusmatrix_on_openness_iset(self):

        for params in ParamGridIter(self.params_range):

            # Loading Vocabulary.
            split_suffix = '_S' + str(params['uknw_ctgs_num']) +\
                '_I' + str(params['uknw_ctgs_num_splt_itrs']) +\
                '_kF' + str(params['kfolds'])
            vocab_fname = self.state_path + 'Vocab' + split_suffix

            with open(vocab_fname + '.pkl', 'r') as f:
                tf_vocab = pickle.load(f)

            # Get the Vocabulary keeping all the terms with same freq to the...
            # ...last feature of the requested size.
            resized_tf_vocab = tfdutils.keep_atleast(tf_vocab, params['vocab_size'])

            # Saving the real Vocabulary sizes for this experiment...
            # ??

            # Creating the Terms-Index Vocabulary that is shorted by Frequency's...
            # ...descending order
            tid_vocab = tfdutils.tf2tidx(resized_tf_vocab)

            corpus_fname = self.state_path + 'Corpus_' +\
                'VS' + str(params['vocab_size']) + split_suffix + '.h5'

            # Creating TF Vectors Matrix (pyTables TF EArray)
            res = self.terms_model.from_files(
                xhtml_file_l=list(self.html_file_l),
                tid_vocabulary=tid_vocab, norm_func=self.norm_func,
                h5_fname=corpus_fname,
                encoding=self.encoding, error_handling=self.encoding
            )

            corpus_mtrx = res[0]
            h5f = res[1]

            # Saving TF Vectors Corpus Matrix
            h5f.close()
            h5f = tb.open_file(corpus_fname, 'r+')
            corpus_mtrx = h5f.get_node('/',  'corpus_earray')

            """
            corpus_fname = self.state_path + 'Corpus_' +\
                'VS' + str(params['vocab_size']) + split_suffix + '.pkl'
            with open(corpus_fname, 'w') as f:
                pickle.dump(corpus_mtrx, f)
            """

    def build_corpusmatrix_on_dlparams(self):

        for params in ParamGridIter(self.params_range):

            # Building the corpus matrix with a specific Normalizing function.
            # NOTE: The corpus here will NOT be normalized.
            print "Building the Corpus Matrix (Tables) GensimVec..."

            split_suffix = '_S' + str(params['uknw_ctgs_num']) +\
                '_I' + str(params['uknw_ctgs_num_splt_itrs']) +\
                '_kF' + str(params['kfolds'])

            corpus_fname = self.state_path + 'Corpus_' +\
                'GDims' + str(params['dims']) +\
                'GParams' + str(params['min_trm_fq']) + '_' +\
                str(params['win_size']) + '_' +\
                str(params['algo']) + '_' +\
                str(params['alpha']) + '_' +\
                str(params['min_alpha']) + '_' +\
                str(params['epochs']) + '_' +\
                str(params['decay']) + '_' +\
                split_suffix + '.h5'

            corpus_mtrx, h5f = self.terms_model.from_files(
                xhtml_file_l=list(self.html_file_l), norm_func=self.norm_func,

                # Specific paramtera for Gensim.
                dims=params['dims'], min_trm_fq=params['min_trm_fq'], win_size=params['win_size'],
                algo=params['algo'], alpha=params['alpha'], min_alpha=params['min_alpha'],
                epochs=params['epochs'], decay=params['decay'],

                h5_fname=corpus_fname, encoding=self.encoding, error_handling=self.encoding
            )

            # Saving TF Vectors Corpus Matrix
            h5f.close()
            h5f = tb.open_file(corpus_fname, 'r+')
            corpus_mtrx = h5f.get_node('/',  'corpus_earray')

    def evaluate_on_openness_deepl(self):

        # Loading the last good states list for skipping the sates which already has been evaluated.
        last_goodstate_lst = list()
        if os.path.exists(self.state_path+'last_good_sate.jsn'):
            with open(self.state_path+'last_good_sate.jsn', 'r') as f:
                last_goodstate_lst = json.load(f)

        # Setting initial value for the variable will be used also for not re-loading a file has...
        # ...been loaded in the exact previous iteration.
        last_splt_fname_suffix = ''

        # Starting Parameters Grid Search
        for gci, params in enumerate(ParamGridIter(self.params_range)):

            # Show how many Grid Search Parameter combinations are renaming.
            print "Param Grid Counts:", gci+1

            print "Params: ", params

            # # # Creating the group sequence respectively to the models parameters:
            # Assigning Feature number group to next_group parameter for initializing the loop
            next_group = self.h5_res.root

            # Start the loop of creating or getting group nodes in respect to model parameters
            for pname, pvalue in params.items():
                try:
                    next_group = self.h5_res.get_node(
                        next_group, pname+str(pvalue).replace('.', '')
                    )
                except Exception as exp_signal:
                    next_group = self.h5_res.create_group(
                        next_group, pname+str(pvalue).replace('.', ''), "<Comment>"
                    )
            # # # END- Group creation sequence

            # Setting initial value for the variable will be used also for not re-loading
            # ...a file has been loaded in the exact previous iteration.
            last_corpus_fname = ''

            # Skipping the states that have already been tested.
            this_state_params = params.values()
            # print last_goodstate_lst
            if this_state_params in last_goodstate_lst:
                print "Skipping already tested state: ", this_state_params
                continue

            # Loading corpus matrix for this Sub-Split.
            split_suffix = '_S' + str(params['uknw_ctgs_num']) +\
                '_I' + str(params['uknw_ctgs_num_splt_itrs']) +\
                '_kF' + str(params['kfolds'])

            corpus_fname = self.state_path + 'Corpus_' +\
                'GDims' + str(params['dims']) +\
                'GParams' + str(params['min_trm_fq']) + '_' +\
                str(params['win_size']) + '_' +\
                str(params['algo']) + '_' +\
                str(params['alpha']) + '_' +\
                str(params['min_alpha']) + '_' +\
                str(params['epochs']) + '_' +\
                str(params['decay']) + '_' +\
                split_suffix + '.pkl'

            # If not already loading the corpus matrix.
            if last_corpus_fname != corpus_fname:

                # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
                corpus_mtrx, file_obj = self.LoadCorpusMatrix(corpus_fname, '/')

                # Selecting Cross Validation Set.
                # Getting the Indeces of samples for each part of the testing sub-split.
                tsp_idxs = test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
                onlysp_idxs = onlyt_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]

                # Getting the full testing-samples class tags, including the original class..
                # ...tags of the only-test classes.
                expected_Y = self.cls_tgs[tsp_idxs]

                # Preplacing with class tags of the sammples which are are belonging to the...
                # ...Only-Test with 0, i.e. as expected to be Unknown a.k.a. "Don't Know"...
                # ...expected predictions.
                expected_Y[np.in1d(tsp_idxs, onlysp_idxs)] = 0

                # Evaluating Semi-Supervised Classification Method.
                print "EVALUATING"
                # predicted_Y, predicted_scores, model_specific_d = self.model(
                #     train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     corpus_mtrx,
                #     self.cls_tgs,
                #     params
                # )

                res_d = self.model(
                    train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                    test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                    expected_Y,
                    corpus_mtrx,
                    self.cls_tgs,
                    params
                )

                # predicted_Y, predicted_d_near, predicted_d_far, gnr_cls_idx = self.model.eval(
                #     train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     expected_Y,
                #     corpus_mtrx,
                #     self.cls_tgs,
                #     params
                # )

                print 'P Y shape:', predicted_Y.shape
                print 'E Y shape:', expected_Y.shape

                'max_sim_scores_per_iter'
                'predicted_classes_per_iter'

                # Saving results
                for rname, rval in res_d.items():

                    self.h5_res.create_array(
                        next_group, rname, rval,
                        ""
                    )

                # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
                if file_obj is not None:
                    file_obj.close()

                # Saving the last good state. Then the process can continue after this state in...
                # ...order not to start every Evaluation again.
                with open(self.state_path+'last_good_sate.jsn', 'w') as f:
                    pram_vals = params.values()
                    last_goodstate_lst.append(pram_vals)
                    json.dump(last_goodstate_lst, f)

        # Return Results H5 File handler class
        return self.h5_res

    def evaluate_on_openness_iset(self):

        if not os.path.exists(self.corps_fpath):
            raise Exception("Corpus files path does not exist.")

        # Loading the last good states list for skipping the sates which already has been evaluated.
        last_goodstate_lst = list()
        if os.path.exists(self.state_path + 'last_good_sate.jsn'):
            with open(self.state_path + 'last_good_sate.jsn', 'r') as f:
                last_goodstate_lst = json.load(f)

        # Setting initial value for the variable will be used also for not re-loading a file has...
        # ...been loaded in the exact previous iteration.
        last_splt_fname_suffix = ''

        # Starting Parameters Grid Search
        for gci, params in enumerate(ParamGridIter(self.params_range)):

            # Show how many Grid Search Parameter combinations are renaming.
            print "Param Grid Counts:", gci+1

            print "Params: ", params

            # # # Creating the group sequence respectively to the models parameters:
            # Assigning Feature number group to next_group parameter for initializing the loop
            next_group = self.h5_res.root

            # Start the loop of creating or getting group nodes in respect to model parameters
            for pname, pvalue in params.items():
                try:
                    next_group = self.h5_res.get_node(
                        next_group, pname+str(pvalue).replace('.', '')
                    )
                except Exception as exp_signal:
                    next_group = self.h5_res.create_group(
                        next_group, pname+str(pvalue).replace('.', ''), "<Comment>"
                    )
            # # # END- Group creation sequence

            # Setting initial value for the variable will be used also for not re-loading
            # ...a file has been loaded in the exact previous iteration.
            last_corpus_fname = ''

            # Skipping the states that have already been tested.
            this_state_params = params.values()
            # print last_goodstate_lst
            if this_state_params in last_goodstate_lst:
                print "Skipping already tested state: ", this_state_params
                continue

            # Loading corpus matrix for this Sub-Split.
            split_suffix = '_S' + str(params['uknw_ctgs_num']) +\
                '_I' + str(params['uknw_ctgs_num_splt_itrs']) +\
                '_kF' + str(params['kfolds'])

            corpus_fname = self.state_path + 'Corpus_' +\
                'VS' + str(params['vocab_size']) + split_suffix

            # If not already loading the corpus matrix.
            if last_corpus_fname != corpus_fname:

                # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
                corpus_mtrx, file_obj = self.LoadCorpusMatrix(corpus_fname, '/')

                # Selecting Cross Validation Set.
                # Getting the Indeces of samples for each part of the testing sub-split.
                tsp_idxs = test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
                onlysp_idxs = onlyt_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]

                # Getting the full testing-samples class tags, including the original class..
                # ...tags of the only-test classes.
                expected_Y = self.cls_tgs[tsp_idxs]

                # Preplacing with class tags of the sammples which are are belonging to the...
                # ...Only-Test with 0, i.e. as expected to be Unknown a.k.a. "Don't Know"...
                # ...expected predictions.
                expected_Y[np.in1d(tsp_idxs, onlysp_idxs)] = 0

                # Evaluating Semi-Supervised Classification Method.
                print "EVALUATING"
                # predicted_Y, predicted_scores, model_specific_d = self.model(
                #     train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     corpus_mtrx,
                #     self.cls_tgs,
                #     params
                # )


                predicted_Y, predicted_R, optimal_RT = self.model(
                    train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                    test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                    corpus_mtrx,
                    self.cls_tgs,
                    params
                )

                # predicted_Y, predicted_d_near, predicted_d_far, gnr_cls_idx = self.model.eval(
                #     train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     expected_Y,
                #     corpus_mtrx,
                #     self.cls_tgs,
                #     params
                # )

                print 'P Y shape:', predicted_Y.shape
                print 'E Y shape:', expected_Y.shape

                'max_sim_scores_per_iter'
                'predicted_classes_per_iter'

                # Saving results
                for rname, rval in res_d.items():

                    self.h5_res.create_array(
                        next_group, rname, rval,
                        ""
                    )

                # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
                if file_obj is not None:
                    file_obj.close()

                # Saving the last good state. Then the process can continue after this state in...
                # ...order not to start every Evaluation again.
                with open(self.state_path+'last_good_sate.jsn', 'w') as f:
                    pram_vals = params.values()
                    last_goodstate_lst.append(pram_vals)
                    json.dump(last_goodstate_lst, f)

                """
                self.h5_res.create_array(
                    next_group, 'expected_Y', expected_Y,
                    ""
                )
                self.h5_res.create_array(
                    next_group, 'predicted_Y', predicted_Y,
                    ""
                )
                self.h5_res.create_array(
                    next_group, 'predicted_R', predicted_R,
                    ""
                )
                self.h5_res.create_array(
                    next_group, 'optimal_RT', np.array([optimal_RT]),
                    ""
                )

                self.h5_res.create_array(
                    next_group, 'predicted_Y', predicted_Y,
                    ""
                )

                self.h5_res.create_array(
                    next_group, 'predicted_scores', predicted_scores,
                    ""
                )

                self.h5_res.create_array(
                    next_group, 'predicted_Ns_per_gnr',  predicted_d_near,
                    ""
                )
                self.h5_res.create_array(
                    next_group, 'predicted_Fs_per_gnr', predicted_d_far,
                    ""
                )
                self.h5_res.create_array(
                    next_group, 'gnr_cls_idx', gnr_cls_idx,
                    ""
                )

                if model_specific_d:
                    for name, value in model_specific_d.items():
                        self.h5_res.create_array(next_group, name, value, "<Comment>")[:]

                """
        # Return Results H5 File handler class
        return self.h5_res

"""
class OpenSetParamGridSearchTables(OpenSetParamGridSearchBase):

    def __init__(self, model, terms_model_module, class_names_lst,
                 h5_file_results, raw_corpus_files_path, process_state_saving_path):

        # Passing the argument to the Super-Class
        super(OpenSetParamGridSearchTables, self).__init__(
            model, terms_model_module, class_names_lst,
            h5_file_results, raw_corpus_files_path, process_state_saving_path
        )

    def SaveCorpusMatrix(self, corpus_mtrx, filename, file_obj, process_state_saving_path=None):

        # Does Nothing. It is only usfull for Numpy/Scipy.sparse Arrays/Matrices.
        process_state_saving_path = None



        return file_obj, corpus_mtrx

    def LoadCorpusMatrix(self, filename, process_state_saving_path=None):

        # Replace the path where the process-state files was supposed to be saved.
        if process_state_saving_path:
            save_path = process_state_saving_path
        else:
            save_path = self.state_path

        if not os.path.exists(save_path):
            raise Exception(
                "Loading Samples Splits Faild: process-state-saving-path does not exist"
            )

        # Setting the filename to load the Corpus Matrix (Spase).
        corpus_mtrx_fname = save_path + filename + '.h5'

        if os.path.exists(corpus_mtrx_fname):

            print "Loading Corpus Matrix..."

            # Loading Coprus Matrix (pyTables TF EArray).
            h5f = tb.open_file(corpus_mtrx_fname, 'r+')

            corpus_mtrx = h5f.get_node('/',  'corpus_GsmDoc2Vec_array')  # h5f.root.corpus_earray

        else:
            return None, None

        return (corpus_mtrx, h5f)
"""
