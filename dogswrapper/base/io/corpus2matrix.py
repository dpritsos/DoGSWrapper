# -*- coding: utf-8 -*-

import json
import os
import cPickle as pickle
import numpy as np
import tables as tb
import scipy.sparse as ssp
import param_combs
import sys
sys.path.append('../../../')
from html2vec.utils import tfdutils
from html2vec.base.io.basefilehandlers import file_list_frmpaths


def BuildCorpusMatrixTables(self, html_file_l, filename, tid_vocab, norm_func, encoding='utf-8'):

    # Does nothing. It is only usefull for pyTables.
    filename = None

    print "Building the Corpus Matrix..."

    # Creating TF Vectors Matrix
    corpus_mtrx = self.terms_tf.from_files(
        xhtml_file_l=html_file_l, tid_vocabulary=tid_vocab,
        norm_func=norm_func, encoding=encoding, error_handling='replace'
    )[0]  # <--- Be careful with zero index

    # Returning the Corpus Matrix aligned upon the given Vocabulary.
    return (corpus_mtrx, filename)


def BuildCorpusMatrix(self, html_file_l,
                      dims, min_trm_fq, win_size, algo,
                      alpha, min_alpha, epochs, decay,
                      h5_fname, tid_vocabulary, norm_func, encoding='utf-8'):

    # Setting the pyTables suffix, just for separating them from Numpy pickled Arrays/Matrices.
    h5_fname = h5_fname + '.h5'

    print "Building the Corpus Matrix (Tables)..."

    # Creating TF Vectors Matrix (pyTables TF EArray)
    corpus_mtrx_gsm, corpus_mtrx_fq, h5f = self.terms_tf.from_files(
        xhtml_file_l=html_file_l,
        dims=dims, min_trm_fq=min_trm_fq, win_size=win_size, algo=algo,
        alpha=alpha, min_alpha=min_alpha, epochs=epochs, decay=decay,
        h5_fname=h5_fname, tid_vocabulary=tid_vocabulary,
        norm_func=norm_func, encoding=encoding, error_handling='replace'
    )[0:3]  # <--- Getting only 2 of the 3 returned values.

    # Returning the Corpus Matrix aligned upon the given Vocabulary.
    return (corpus_mtrx_gsm, h5f)



def SaveCorpusMatrix(self, corpus_mtrx, filename, file_obj, process_state_saving_path=None):

    # Does nothing. It is only usefull for pyTables.
    file_obj = None

    # Replace the default or create if required the path where Corpus Matrix will be saved.
    if process_state_saving_path:
        save_path = process_state_saving_path
    else:
        save_path = self.state_save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Set the file names for the Corpus matrix.
    corpus_mtrx_fname = save_path + filename + '.pkl'

    # Saving TF Vectors Corpus Matrix
    print "Saving the Corpus TF Matrix..."
    with open(corpus_mtrx_fname, 'w') as f:
        pickle.dump(corpus_mtrx, f)

    return None, corpus_mtrx

def LoadCorpusMatrix(self, filename, process_state_saving_path=None):

    # Replace the path where the process-state files was supposed to be saved.
    if process_state_saving_path:
        save_path = process_state_saving_path
    else:
        save_path = self.state_save_path

    if not os.path.exists(save_path):
        raise Exception(
            "Loading Samples Splits Faild: process-state-saving-path does not exist"
        )

    # Setting the filename to load the Corpus Matrix (Spase).
    corpus_mtrx_fname = save_path + filename + '.pkl'

    if os.path.exists(corpus_mtrx_fname):

        print "Loading Corpus Matrix..."

        # Loading Coprus Matrix.
        with open(corpus_mtrx_fname, 'r') as f:
            corpus_mtrx = pickle.load(f)

    else:
        return None, None

    return (corpus_mtrx, None)





class OpenSetParamGridSearchBase(object):

    def __init__(self, model, terms_tf_module, class_names_lst,
                 h5_file_results, raw_corpus_files_path, process_state_saving_path):

        self.model = model
        self.terms_tf = terms_tf_module
        self.classes_lst = class_names_lst
        self.classes_num = len(class_names_lst)
        self.h5_res = h5_file_results
        # if not os.path.exists(self.h5_res):
        #    raise Exception("Results HD5 file for saving results does not exist.")

        self.corpus_files_path = raw_corpus_files_path
        if not os.path.exists(self.corpus_files_path):
            raise Exception("Corpus files path does not exist.")

        self.state_save_path = process_state_saving_path
        if not os.path.exists(self.state_save_path):
            print "Creating a new process-state save path..."
            os.mkdir(self.state_save_path)
            print "New process-state saving path is: '" + self.state_save_path + "'"

    def LoadCrpsFnamesTags(self, raw_corpus_files_path=None, process_state_saving_path=None):

        # Replace the class instantiation defined variable self.corpus_files_path if any.
        if raw_corpus_files_path:
            self.corpus_files_path = raw_corpus_files_path

        if not os.path.exists(self.corpus_files_path):
            raise Exception("Corpus files path does not exist.")

        # Replace and create if required the path where the process-state files will be saved.
        if process_state_saving_path:
            self.state_save_path = process_state_saving_path

        if not os.path.exists(self.state_save_path):
            os.mkdir(self.state_save_path)
            print "New process-state saving path is: '" + self.state_save_path + "'"

        # Set the file names for the Filenames and Tags lists to be Loaded or Saved.
        corpus_files_lst_path = self.state_save_path + '/Corpus_filenames_sorted.lst'
        corpus_tags_lst_path = self.state_save_path + '/Corpus_tags_sorted.lst'

        if os.path.exists(corpus_files_lst_path) and os.path.exists(corpus_tags_lst_path):

            print "Loading HTML Filenames list and Classes Tags array..."

            # Load Filename and classes Tags.
            with open(corpus_files_lst_path, 'r') as f:
                html_file_l = json.load(f, encoding='utf-8')

            with open(corpus_tags_lst_path, 'r') as f:
                cls_tgs = json.load(f, encoding='utf-8')

        else:

            print "Loading HTML Filenames from given a file path..."

            html_file_l = list()
            cls_tgs = list()

            # Get the list of Genre argument as given to this Class and build html-file-list...
            # ...and class-genres-tags list
            for i, g in enumerate(self.classes_lst):

                # Get all files located to the genre's path 'g'
                gnrs_file_lst = file_list_frmpaths(self.corpus_files_path, [str(g + "/html/")])

                # Extends the list of html files with the set of files form genre 'g'.
                html_file_l.extend(gnrs_file_lst)

                # Extends the list of html files with the set of class tag form genre 'g',...
                # ...i.e. the index of the genre's list given as argument to this class...
                # ...(ParamGridCrossValBase()).
                cls_tgs.extend([i+1]*len(gnrs_file_lst))

            print "Saving Filenames list and Classes Tags Numpy array..."

            # Saving Filename and classes Tags lists
            with open(corpus_files_lst_path, 'w') as f:
                # HTML File List as founded in the Ext4 file system by python built-it os...
                # ...(python 2.7.x) lib
                json.dump(html_file_l, f, encoding='utf-8')

            with open(corpus_tags_lst_path, 'w') as f:
                # Assigned Genre Tags to files list Array
                json.dump(cls_tgs, f, encoding='utf-8')

        print "Returning the Corpus Filenames list and Classes Tags Numpy array."

        # Returning the filename list and the tags array.
        return np.array(html_file_l), np.array(cls_tgs)



    def MaxNormalise(self, corpus_mtrx, vocab_len):

        # Getting the Maximum frequency for every document.
        max_val = np.max(corpus_mtrx.todense())

        if max_val == 0.0:

            # NOTE: Preventing division-by-zero For Documents with zero terms. This case occurs...
            # when a sub-Vocabulary is used for the experiment.
            max_val = 1.0

            # NOTE: PATCH for preventing All-Zero-Values vectors stopping the experiments.
            corpus_mtrx[:] = 1e-15

        # Normalizing based on the matrix/array type.
        if ssp.issparse(corpus_mtrx):
            corpus_mtrx = ssp.csr_matrix(corpus_mtrx.todense() / max_val)
        else:
            corpus_mtrx = corpus_mtrx / max_val

        return corpus_mtrx

    def SubSamplingNorm(self, corpus_mtrx, vocab_len, sb_t=0.0001):

        # Getting the Maximum frequency for every document.
        max_val = np.max(corpus_mtrx.todense())

        # NOTE: PATCH for preventing All-Zero-Values vectors stopping the experiments.
        corpus_mtrx[np.where(corpus_mtrx == 0.0)] = 1e-15

        # Applying SubSampling with pre Max-Normalization.
        if ssp.issparse(corpus_mtrx):
            corpus_mtrx = ssp.csr_matrix(1 - np.sqrt(sb_t / (corpus_mtrx.todense() / max_val)))
        else:
            corpus_mtrx = 1 - np.sqrt(sb_t / (corpus_mtrx / max_val))

        return corpus_mtrx

    def EvaluateAll(self, params_range, raw_corpus_files_path=None, encoding='utf-8'):
        """
            Parameters Template
            -------------------
            params_range = coll.OrderedDict([
               ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               ('uknw_ctgs_num', [1, 2, 3, 4, 5]),
               ('uknw_ctgs_num_splt_itrs', [0, 1, 2, 3, 4, 5, 6]),
               ('vocab_size', [5000, 10000, 50000, 100000]),
               ('features_size', [500, 1000, 5000, 10000, 50000, 90000]),
               ('Sigma', [0.5, 0.7, 0.9]),
               ('Iterations', [10, 50, 100])
            ])

            params_range = coll.OrderedDict([
                ('vocab_size', [100000]),
                # ('features_size', [10000]),
                ('split_ptg', [0.2]),
                ('ukwn_slt_ptg', [0.5]),
                ('rt_lims_stp', [[0.3, 0.9, 0.2]]),
                ('lmda', [0.5]),
                ('uknw_ctgs_num', [1]),
                ('uknw_ctgs_num_splt_itrs', [0]),
                ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ])

        """

        # Replace the class instantiation defined variable self.corpus_files_path if any.
        if raw_corpus_files_path:
            self.corpus_files_path = raw_corpus_files_path

        if not os.path.exists(self.corpus_files_path):
            raise Exception("Corpus files path does not exist.")

        # Loading the Filename list of the corpus and their respective class tags.
        html_file_l, cls_tgs = self.LoadCrpsFnamesTags()

        # Loading the last good states list for skipping the sates which already has been evaluated.
        last_goodstate_lst = list()
        if os.path.exists(self.state_save_path+'last_good_sate.jsn'):
            with open(self.state_save_path+'last_good_sate.jsn', 'r') as f:
                last_goodstate_lst = json.load(f)

        # Setting initial value for the variable will be used also for not re-loading a file has...
        # ...been loaded in the exact previous iteration.
        last_splt_fname_suffix = ''

        # Starting Parameters Grid Search
        for gci, params in enumerate(param_combs.ParamGridIter(params_range)):

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
                except:
                    next_group = self.h5_res.create_group(
                        next_group, pname+str(pvalue).replace('.', ''), "<Comment>"
                    )
            # # # END- Group creation sequence

            # Forming the Training/Testing Splits filename suffix. If it is the same with the...
            # ...previous iteration's one just skip the file loading, because it is already there.
            splt_fname_suffix = '_S' + str(params['uknw_ctgs_num']) + '_I' +\
                str(len(params_range['uknw_ctgs_num_splt_itrs']))

            if last_splt_fname_suffix != splt_fname_suffix:

                trn_fname = self.state_save_path + 'Training_Splits' + splt_fname_suffix + '.pkl'
                test_fname = self.state_save_path + 'Testing_Splits' + splt_fname_suffix + '.pkl'
                oTest_fname = self.state_save_path + 'OnlyTesting_Splits' +\
                    splt_fname_suffix + '.pkl'

                # Loading Training/Testing Splits.
                train_splts, test_splts, onlyt_splts = self.LoadSplitSamples(
                    (trn_fname, test_fname, oTest_fname), '/'
                )

                # In case 'None' has been loaded: Building and saving splits upon params for...
                # ...the next iteration will be needed.
                if not (train_splts and test_splts and onlyt_splts):

                    # Building the splits.
                    train_splts, test_splts, onlyt_splts = self.OpennessSplitSamples(
                        cls_tgs,
                        onlytest_clsnum=params['uknw_ctgs_num'],
                        uknw_ctgs_num_splt_itrs=len(params_range['uknw_ctgs_num_splt_itrs']),
                        kfolds=len(params_range['kfolds'])
                    )

                    # Saving the splits.
                    self.SaveSplitSamples(
                        train_splts, test_splts, onlyt_splts,
                        (trn_fname, test_fname, oTest_fname), '/'
                    )

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

            corpus_fname = self.state_save_path + 'Corpus_' +\
                'VS' + str(params['vocab_size']) + split_suffix

            # If not already loading the corpus matrix.
            if last_corpus_fname != corpus_fname:

                # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
                corpus_mtrx, file_obj = self.LoadCorpusMatrix(corpus_fname, '/')

                # If 'None' corpus matrix has been loaded build it.
                if corpus_mtrx is None:

                    vocab_fname = self.state_save_path + 'Vocab' + split_suffix

                    # Loading the proper Vocabulary.
                    if os.path.exists(vocab_fname+'.pkl'):

                        # Loading the vocabulary.
                        print "Loading Vocabulary..."
                        with open(vocab_fname+'.pkl', 'r') as f:
                            tf_vocab = pickle.load(f)

                    else:
                        # Building the Vocabulary if not already exists.
                        print "Building Vocabulary..."

                        # Building the TF Vocabulary.
                        tf_vocab = self.terms_tf.build_vocabulary(
                            list(
                                html_file_l[
                                    train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
                                ]
                            ),
                            encoding=encoding, error_handling='replace'
                        )

                        # Saving TF Vocabulary in pickle and Json format.
                        print "Saving Vocabulary..."
                        with open(vocab_fname+'.pkl', 'w') as f:
                            pickle.dump(tf_vocab, f)

                        with open(vocab_fname+'.jsn', 'w') as f:
                            json.dump(tf_vocab, f, encoding=encoding)

                    # Get the Vocabulary keeping all the terms with same freq to the...
                    # ...last feature of the requested size.
                    resized_tf_vocab = tfdutils.keep_atleast(tf_vocab, params['vocab_size'])

                    # Saving the real Vocabulary sizes for this experiment...
                    # ...(i.e. this text representation, etc.) keep it as pytables group...
                    # ...attribute the actual Vocabulary size.

                    # DO I NEED IT?
                    # vocab_size_group._v_attrs.real_voc_size = [(k, len(resized_tf_vocab))]

                    # Creating the Terms-Index Vocabulary that is shorted by Frequency's...
                    # ...descending order
                    tid_vocab = tfdutils.tf2tidx(resized_tf_vocab)

                    # Building the corpus matrix with a specific Normalizing function.
                    # NOTE: The corpus is max-normalized.
                    print 'Building Corpus Matrix...'

                    corpus_mtrx, file_obj = self.BuildCorpusMatrix(
                        list(html_file_l),
                        dims=100, min_trm_fq=0, win_size=8, algo='PV-DM',
                        alpha=0.025, min_alpha=0.025, epochs=10, decay=0.002,
                        h5_fname=corpus_fname, tid_vocabulary=tid_vocab,
                        norm_func=None, encoding=encoding
                    )

                    # NOTE: Saving the corpus matrix in normalized form.
                    file_obj, corpus_mtrx = self.SaveCorpusMatrix(
                        corpus_mtrx, corpus_fname, file_obj, '/'
                    )

                # Selecting Cross Validation Set.
                # Getting the Indeces of samples for each part of the testing sub-split.
                tsp_idxs = test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]
                onlysp_idxs = onlyt_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']]

                # Getting the full testing-samples class tags, including the original class..
                # ...tags of the only-test classes.
                expected_Y = cls_tgs[tsp_idxs]

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
                #     cls_tgs,
                #     params
                # )

                predicted_Y, predicted_R, optimal_RT = self.model(
                    train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                    test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                    corpus_mtrx,
                    cls_tgs,
                    params
                )

                # predicted_Y, predicted_d_near, predicted_d_far, gnr_cls_idx = self.model.eval(
                #     train_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     test_splts[params['uknw_ctgs_num_splt_itrs']][params['kfolds']],
                #     expected_Y,
                #     corpus_mtrx,
                #     cls_tgs,
                #     params
                # )

                print 'P Y shape:', predicted_Y.shape
                print 'E Y shape:', expected_Y.shape

                'max_sim_scores_per_iter'
                'predicted_classes_per_iter'

                # Saving results

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

                """
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

                # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
                if file_obj is not None:
                    file_obj.close()

                # Saving the last good state. Then the process can continue after this state in...
                # ...order not to start every Evaluation again.
                with open(self.state_save_path+'last_good_sate.jsn', 'w') as f:
                    pram_vals = params.values()
                    last_goodstate_lst.append(pram_vals)
                    json.dump(last_goodstate_lst, f)

        # Return Results H5 File handler class
        return self.h5_res


class OpenSetParamGridSearchTables(OpenSetParamGridSearchBase):

    def __init__(self, model, terms_tf_module, class_names_lst,
                 h5_file_results, raw_corpus_files_path, process_state_saving_path):

        # Passing the argument to the Super-Class
        super(OpenSetParamGridSearchTables, self).__init__(
            model, terms_tf_module, class_names_lst,
            h5_file_results, raw_corpus_files_path, process_state_saving_path
        )



    def SaveCorpusMatrix(self, corpus_mtrx, filename, file_obj, process_state_saving_path=None):

        # Does Nothing. It is only usfull for Numpy/Scipy.sparse Arrays/Matrices.
        process_state_saving_path = None

        # Closing and re-opening file just for safety.
        file_obj.close()
        file_obj = tb.open_file(filename + '.h5', 'r+')
        corpus_mtrx = file_obj.get_node('/',  'corpus_GsmDoc2Vec_array')

        return file_obj, corpus_mtrx

    def LoadCorpusMatrix(self, filename, process_state_saving_path=None):

        # Replace the path where the process-state files was supposed to be saved.
        if process_state_saving_path:
            save_path = process_state_saving_path
        else:
            save_path = self.state_save_path

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
