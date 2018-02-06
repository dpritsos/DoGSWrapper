# -*- coding: utf-8 -*-

import csv
import json
import os
import cPickle as pickle
import numpy as np
import tables as tb
import scipy.sparse as ssp
import collections as coll

from ..tools.loaders import LoadCrpsFnamesTags
from ..tools.samplers import OpennessSplitSamples
from ..tools.samplers import LoadSplitSamples, SaveSplitSamples
from ..tools.paramcombs import ParamGridIter

import time as tm

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

        # Loading State file and Skipping ISet creation if already done.
        this_state = ['Openness Index Splits/Folds Sets - Created']
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))
                if this_state in last_goodstate:
                    print "Skipping Index Splits/Folds Sets creation"
                    return

        # NOTE: Selecting the required parameter ranges for this process only.
        selected_params = coll.OrderedDict([
            ('uknw_ctgs_num', self.params_range['uknw_ctgs_num']),
        ])

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])
        kfolds = len(self.params_range['kfolds'])

        # Creating the Openness Kfold-Splits.
        # NOTE: KFolds here are given as a count not as part of Param Grid Iteratable sets.
        for params in ParamGridIter(selected_params):

            # Building the splits.
            train_splts, test_splts, onlyt_splts = OpennessSplitSamples(
                self.cls_tgs,
                onlytest_clsnum=params['uknw_ctgs_num'],
                uknw_ctgs_num_splt_itrs=ukn_iters,
                kfolds=kfolds
            )

            splt_fname_suffix = '_S' + str(params['uknw_ctgs_num']) + '_I' + str(ukn_iters)
            SaveSplitSamples(
                train_splts, test_splts, onlyt_splts, splt_fname_suffix, self.state_path
            )

        # Saving the last good state.
        with open(self.state_path + 'last_good_sate.csv', 'a') as f:
            cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            cwriter.writerow(this_state)

    def build_vocabulary_on_openness_iset(self):

        # Building the Vocabularies for all Splits/Folds
        print "Building Vocabularies..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        # NOTE: Selecting the required parameter ranges for this process only.
        selected_params = coll.OrderedDict([
            ('uknw_ctgs_num', self.params_range['uknw_ctgs_num']),
            ('uknw_ctgs_num_splt_itrs', self.params_range['uknw_ctgs_num_splt_itrs']),
            ('kfolds', self.params_range['kfolds']),
        ])

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])

        for params in ParamGridIter(selected_params):

            # Skipping the Creating of this Vocabulary if alaready created.
            this_state = ['Vocabulary for: ' + str(params) + ' - Created']
            if this_state in last_goodstate:
                print "Skipping Vocabulary creation for: " + str(params)
                continue

            splt_fname_suffix = '_S' + str(params['uknw_ctgs_num']) + '_I' + str(ukn_iters)
            train_splts, test_splts, onlyt_splts = LoadSplitSamples(
                splt_fname_suffix, self.state_path
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
                json.dump(tf_vocab, f)

            # Saving the last good state.
            with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                cwriter.writerow(this_state)

    def build_corpusmatrix_on_openness_iset(self):

        # Building the Vocabularies for all Splits/Folds
        print "Corpus Matrices..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        # NOTE: Selecting the required parameter ranges for this process only.
        selected_params = coll.OrderedDict([
            ('vocab_size', self.params_range['vocab_size']),
            ('uknw_ctgs_num', self.params_range['uknw_ctgs_num']),
            ('uknw_ctgs_num_splt_itrs', self.params_range['uknw_ctgs_num_splt_itrs']),
            ('kfolds', self.params_range['kfolds']),
        ])

        for params in ParamGridIter(selected_params):

            # Skipping the Creating of this Corpus Matrix if alaready created.
            this_state = ['Corpus Matrix for: ' + str(params) + ' - Created']
            if this_state in last_goodstate:
                print "Skipping Corpus Matrix creation for: " + str(params)
                continue

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
                encoding=self.encoding, error_handling=self.error_handling
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

            # Saving the last good state.
            with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                cwriter.writerow(this_state)

    def build_corpusmatrix_on_dlparams(self):

        # Building the Vocabularies for all Splits/Folds
        print "Corpus Matrices..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        # NOTE: Selecting the required parameter ranges for this process only.
        selected_params = coll.OrderedDict([
            ('dims', self.params_range['dims']),
            ('min_trm_fq', self.params_range['min_trm_fq']),
            ('win_size', self.params_range['win_size']),
            ('algo', self.params_range['algo']),
            ('alpha', self.params_range['alpha']),
            ('min_alpha', self.params_range['min_alpha']),
            ('epochs', self.params_range['epochs']),
            ('decay', self.params_range['decay']),
        ])

        for params in ParamGridIter(selected_params):

            # Skipping the Creating of this Corpus Matrix if alaready created.
            this_state = ['Corpus Matrix for: ' + str(params) + ' - Created']
            if this_state in last_goodstate:
                print "Skipping Corpus Matrix creation for: " + str(params)
                continue

            # Building the corpus matrix with a specific Normalizing function.
            # NOTE: The corpus here will NOT be normalized.
            print "Building the Corpus Matrix (Tables) GensimVec..."

            corpus_fname = self.state_path + 'Corpus_' +\
                'GDims' + str(params['dims']) +\
                'GParams' + str(params['min_trm_fq']) + '_' +\
                str(params['win_size']) + '_' +\
                str(params['algo']) + '_' +\
                str(params['alpha']) + '_' +\
                str(params['min_alpha']) + '_' +\
                str(params['epochs']) + '_' +\
                str(params['decay']) + '.h5'

            res = self.terms_model.from_files(
                xhtml_file_l=list(self.html_file_l), norm_func=self.norm_func,

                # Specific paramtera for Gensim.
                dims=params['dims'], min_trm_fq=params['min_trm_fq'], win_size=params['win_size'],
                algo=params['algo'], alpha=params['alpha'], min_alpha=params['min_alpha'],
                epochs=params['epochs'], decay=params['decay'],

                trn_idxs=[None],

                h5_fname=corpus_fname, encoding=self.encoding, error_handling=self.error_handling
            )

            corpus_mtrx = res[0]
            h5f = res[1]

            # Saving TF Vectors Corpus Matrix
            h5f.close()
            h5f = tb.open_file(corpus_fname, 'r+')
            corpus_mtrx = h5f.get_node('/',  'corpus_earray')

            # Saving the last good state.
            with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                cwriter.writerow(this_state)

    def build_corpusmatrix_on_dlparams_per_gnr(self):

        # Building the Vocabularies for all Splits/Folds
        print "Corpus Matrices..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        # NOTE: Selecting the required parameter ranges for this process only.
        selected_params = coll.OrderedDict([
            ('dims', self.params_range['dims']),
            ('min_trm_fq', self.params_range['min_trm_fq']),
            ('win_size', self.params_range['win_size']),
            ('algo', self.params_range['algo']),
            ('alpha', self.params_range['alpha']),
            ('min_alpha', self.params_range['min_alpha']),
            ('epochs', self.params_range['epochs']),
            ('decay', self.params_range['decay']),
        ])

        # Getting unique Class tags and Create a Corpus matrix of Document Models for...
        # ...the whole Corpus but infered only on one and only one Documents Class.
        for ctg in np.unique(self.cls_tgs):

            for params in ParamGridIter(selected_params):

                # Skipping the Creating of this Corpus Matrix if alaready created.
                this_state = ['Corpus Matrix CTG for: ' + str(params) + ' - Created']
                if this_state in last_goodstate:
                    print "Skipping Corpus Matrix creation for: " + str(params)
                    continue

                # Building the corpus matrix with a specific Normalizing function.
                # NOTE: The corpus here will NOT be normalized.
                print "Building the Corpus Matrix (Tables) GensimVec..."

                corpus_fname = self.state_path + 'Corpus_' +\
                    'GDims' + str(params['dims']) +\
                    'GParams' + str(params['min_trm_fq']) + '_' +\
                    str(params['win_size']) + '_' +\
                    str(params['algo']) + '_' +\
                    str(params['alpha']) + '_' +\
                    str(params['min_alpha']) + '_' +\
                    str(params['epochs']) + '_' +\
                    str(params['decay']) + '_' +\
                    'CTG' + str(ctg) + '.h5'

                res = self.terms_model.from_files(
                    xhtml_file_l=list(self.html_file_l), norm_func=self.norm_func,

                    # Specific paramtera for Gensim.
                    dims=params['dims'], min_trm_fq=params['min_trm_fq'],
                    win_size=params['win_size'],
                    algo=params['algo'], alpha=params['alpha'], min_alpha=params['min_alpha'],
                    epochs=params['epochs'], decay=params['decay'],

                    trn_idxs=self.cls_tgs[np.where((self.cls_tgs == ctg))],

                    h5_fname=corpus_fname, encoding=self.encoding,
                    error_handling=self.error_handling
                )

                corpus_mtrx = res[0]
                h5f = res[1]

                # Saving TF Vectors Corpus Matrix
                h5f.close()
                h5f = tb.open_file(corpus_fname, 'r+')
                corpus_mtrx = h5f.get_node('/',  'corpus_earray')

                # Saving the last good state.
                with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                    cwriter = csv.writer(
                        f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL
                    )
                    cwriter.writerow(this_state)

    def evaluate_on_openness_dlparams(self):

        print "EVALUATING..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])

        # Starting Parameters Grid Search
        for gci, params in enumerate(ParamGridIter(self.params_range)):

            ############################################################
            # Works only in for RFSE and OCSVME.
            if params['features_size'] > params['dims']:
                continue
            ############################################################

            # Skipping the Evaluation for this Parameters Set.
            this_state = ['Evaluation for: ' + str(params) + '- Done']
            if this_state in last_goodstate:
                print "Skipping Classification Model Evaluation for: " + str(params)
                continue

            # Show how many Grid Search Parameter combinations are renaming.
            print "Param Grid Counts: ", gci + 1
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
                str(params['decay']) + '.h5'

            # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
            h5f = tb.open_file(corpus_fname, 'r+')
            corpus_mtrx = h5f.get_node('/',  'corpus_earray')

            # Selecting Cross Validation Set.
            # Getting the Indeces of samples for each part of the testing sub-split.
            splt_fname_suffix = '_S' + str(params['uknw_ctgs_num']) + '_I' + str(ukn_iters)
            train_splts, test_splts, onlyt_splts = LoadSplitSamples(
                splt_fname_suffix, self.state_path
            )

            tsp_idxs = test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
            onlysp_idxs = onlyt_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]

            # Getting the full testing-samples class tags, including the original class..
            # ...tags of the only-test classes.
            expected_Y = self.cls_tgs[tsp_idxs]

            # Preplacing with class tags of the sammples which are are belonging to the...
            # ...Only-Test with 0, i.e. as expected to be Unknown a.k.a. "Don't Know"...
            # ...expected predictions.
            expected_Y[np.in1d(tsp_idxs, onlysp_idxs)] = 0

            # NOTE NOTE NOTE
            self.h5_res.create_array(next_group, 'expected_Y', expected_Y, "")

            # TIME IT
            # start_tm = tm.time()

            # Evaluating Semi-Supervised Classification Method.
            print "EVALUATING"
            res_d = self.model.eval(
                train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                # expected_Y,
                corpus_mtrx,
                self.cls_tgs,
                params
            )

            # TIME IT
            # timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            # print "Time elapsed : %d:%d:%d:%d" % timel
            # print

            # print 'P Y shape:', res_d['predicted_Y'].shape
            # print 'E Y shape:', expected_Y.shape

            'max_sim_scores_per_iter'
            'predicted_classes_per_iter'

            # Saving results
            for rname, rval in res_d.items():
                print rname, rval
                self.h5_res.create_array(next_group, rname, rval, "")

            # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
            h5f.close()

            # Saving the last good state.
            with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                cwriter.writerow(this_state)

        # Return Results H5 File handler class
        return self.h5_res

    def evaluate_on_openness_dlparams_per_gnr(self):

        print "EVALUATING..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])

        # Starting Parameters Grid Search
        for gci, params in enumerate(ParamGridIter(self.params_range)):

            # Skipping the Evaluation for this Parameters Set.
            this_state = ['Evaluation for: ' + str(params) + '- Done']
            if this_state in last_goodstate:
                print "Skipping Classification Model Evaluation for: " + str(params)
                continue

            # Show how many Grid Search Parameter combinations are renaming.
            print "Param Grid Counts: ", gci + 1
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

            # Loading corpus matrix for this Sub-Split.
            corpus_mtrx_lst = list()
            for ctg in np.unique(self.cls_tgs):

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
                    'CTG' + str(ctg) + '.h5'

                # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
                h5f = tb.open_file(corpus_fname, 'r+')

                # NOTE NOTE
                corpus_mtrx_lst.append(h5f.get_node('/',  'corpus_earray'))

            # Selecting Cross Validation Set.
            # Getting the Indeces of samples for each part of the testing sub-split.
            splt_fname_suffix = '_S' + str(params['uknw_ctgs_num']) + '_I' + str(ukn_iters)
            train_splts, test_splts, onlyt_splts = LoadSplitSamples(
                splt_fname_suffix, self.state_path
            )

            tsp_idxs = test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
            onlysp_idxs = onlyt_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]

            # Getting the full testing-samples class tags, including the original class..
            # ...tags of the only-test classes.
            expected_Y = self.cls_tgs[tsp_idxs]

            # Preplacing with class tags of the sammples which are are belonging to the...
            # ...Only-Test with 0, i.e. as expected to be Unknown a.k.a. "Don't Know"...
            # ...expected predictions.
            expected_Y[np.in1d(tsp_idxs, onlysp_idxs)] = 0

            # NOTE NOTE NOTE
            self.h5_res.create_array(next_group, 'expected_Y', expected_Y, "")

            # Evaluating Semi-Supervised Classification Method.
            print "EVALUATING"
            res_d = self.model.eval(
                train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                # expected_Y,
                corpus_mtrx_lst,
                self.cls_tgs,
                params
            )

            print 'P Y shape:', res_d['predicted_Y'].shape
            print 'E Y shape:', expected_Y.shape

            'max_sim_scores_per_iter'
            'predicted_classes_per_iter'

            # Saving results
            for rname, rval in res_d.items():
                self.h5_res.create_array(next_group, rname, rval, "")

            # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
            h5f.close()

            # Saving the last good state.
            with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                cwriter.writerow(this_state)

        # Return Results H5 File handler class
        return self.h5_res

    def evaluate_on_openness_iset(self):

        print "EVALUATING..."

        # Loading the last good State.
        if os.path.exists(self.state_path + 'last_good_sate.csv'):
            with open(self.state_path + 'last_good_sate.csv', 'r') as f:
                last_goodstate = list(csv.reader(f, delimiter='\n', quotechar='"'))

        ukn_iters = len(self.params_range['uknw_ctgs_num_splt_itrs'])

        # Starting Parameters Grid Search
        for gci, params in enumerate(ParamGridIter(self.params_range)):

            # Skipping the Evaluation for this Parameters Set.
            this_state = ['Evaluation for: ' + str(params) + '- Done']
            if this_state in last_goodstate:
                print "Skipping Evaluation for: " + str(params)
                continue

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

            # Loading corpus matrix for this Sub-Split.
            split_suffix = '_S' + str(params['uknw_ctgs_num']) +\
                '_I' + str(params['uknw_ctgs_num_splt_itrs']) +\
                '_kF' + str(params['kfolds'])

            corpus_fname = self.state_path + 'Corpus_' +\
                'VS' + str(params['vocab_size']) + split_suffix + '.h5'

            # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
            h5f = tb.open_file(corpus_fname, 'r+')
            corpus_mtrx = h5f.get_node('/',  'corpus_earray')

            # Selecting Cross Validation Set.
            # Getting the Indeces of samples for each part of the testing sub-split.
            splt_fname_suffix = '_S' + str(params['uknw_ctgs_num']) + '_I' + str(ukn_iters)
            train_splts, test_splts, onlyt_splts = LoadSplitSamples(
                splt_fname_suffix, self.state_path
            )

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
            res_d = self.model.eval(
                train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                corpus_mtrx,
                self.cls_tgs,
                params
            )

            # predicted_Y, predicted_d_near, predicted_d_far, gnr_cls_idx = self.model.eval(
            #     train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
            #     test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
            #     expected_Y, # <------------------------------
            #     corpus_mtrx,
            #     self.cls_tgs,
            #     params
            # )

            print 'P Y shape:', res_d['predicted_Y'].shape
            print 'E Y shape:', expected_Y.shape

            'max_sim_scores_per_iter'
            'predicted_classes_per_iter'

            # Saving results
            for rname, rval in res_d.items():
                self.h5_res.create_array(next_group, rname, rval, "")

            self.h5_res.create_array(next_group, 'expected_Y', expected_Y)

            # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
            h5f.close()

            # Saving the last good state. Then the process can continue after this state in...
            # ...order not to start every Evaluation again.
            with open(self.state_path + 'last_good_sate.csv', 'a') as f:
                cwriter = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                cwriter.writerow(this_state)

        # Return Results H5 File handler class
        return self.h5_res
