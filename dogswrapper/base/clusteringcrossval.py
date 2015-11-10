# -*- coding: utf-8 -*-

import json
import os
import cPickle as pickle
import numpy as np
import tables as tb
import numexpr as ne
import scipy.sparse as ssp
import scipy.spatial.distance as spd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn import cross_validation
import param_combs

import sys
sys.path.append('../../html2vectors/')
from html2vect.utils import tfdutils
from html2vect.base.io.basefilehandlers import file_list_frmpaths


class SemiSupervisedParamGridSearchBase(object):

    def __init__(self, semisupervised_model, terms_tf_module, class_names_lst,
                 h5_file_results, raw_corpus_files_path, process_state_saving_path):

        self.semisuper_model = semisupervised_model
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

    def SplitSamples(self, cls_tgs,
                     trn_percent=0.5, decrease_step=0.1, method='rndred-trn-fixed-test'):

        # Checking trn_percent and decrease_step value constraints.
        if trn_percent < 0.01 or trn_percent > 1.0 or decrease_step < 0.01 or decrease_step > 1.0:
            raise Exception("trm_percent and decrease_step values mast be in range [0.01, 1]")

        # Two list of arrays where each array has the file indeces for training and testing...
        # ...repspectivly splitted initially upon trn_percentage.
        trn_splts_per_ctg_arrlst, tst_splts_per_ctg_arrlst = list(), list()

        for ctg in np.unique(cls_tgs):

            # Getting the filename list indeces for this class (tag).
            this_cls_idxs = np.where(cls_tgs == ctg)[0]

            # Calculating the amount of samples keeping for training for this class for the...
            # ...initial split.
            smpls_num = int(np.ceil(this_cls_idxs.shape[0] * trn_percent))

            # NOTE: Here the training indeces are selected Randomly! Thus, it is easy to...
            # ...use this python-class-method into Cross-Validation experimental set-up....
            # ...The list of indeces is sorted.
            train_files_idxs_arr = np.sort(
                np.random.choice(this_cls_idxs, smpls_num, replace=False)
            )

            trn_splts_per_ctg_arrlst.append(train_files_idxs_arr)

            # Keeping the indeces but the ones for training as testing indeces.
            tst_splts_per_ctg_arrlst.append(
                np.short(
                    np.array(
                        [tst_i for tst_i in this_cls_idxs if tst_i not in train_files_idxs_arr]
                    )
                )
            )

        # Two lists per sub-split one for training and one for testing. Every element of the...
        # ...list is containing an array where the rows are containing the training and...
        # ...testing index splits for every class (tag) respectively.
        train_subsplits_arrlst, testing_subsplits_arrlst = list(), list()

        for trn_decreased_perc in np.arange(trn_percent, 0.0, -decrease_step):

            train_ctg_lst, test_ctg_lst = list(), list()

            for trn_arr, tst_arr in zip(trn_splts_per_ctg_arrlst, tst_splts_per_ctg_arrlst):

                smpls_num = int(
                    np.ceil(this_cls_idxs.shape[0] * trn_decreased_perc)
                )

                # Selecting the method to split the corpus to training and test sets.
                if method == 'rndred_trn_fixed_test':

                    # Keeping only a partition of the training indeces split, while the...
                    # ...testning split remains the same.
                    train_ctg_lst.append(trn_arr[0:smpls_num])
                    test_ctg_lst.append(tst_arr)

                elif method == 'rndred_trn_rest4_test':

                    # Keeping only a partition of the training indeces split, while the...
                    # ...testing split is extended with the rest of the training split.
                    train_ctg_lst.append(trn_arr[0:smpls_num])
                    test_ctg_lst.append(
                        np.short(
                            np.hstack(
                                (tst_arr, trn_arr[smpls_num::])
                            )
                        )
                    )

                else:
                    raise Exception("Non-implemented yet!")

            # Keeping the sub-splits array lists.
            train_subsplits_arrlst.append(
                np.vstack(train_ctg_lst)
            )

            testing_subsplits_arrlst.append(
                np.vstack(test_ctg_lst)
            )

        return train_subsplits_arrlst, testing_subsplits_arrlst

    def SaveSplitSamples(self, train_subsplits_arrlst, testing_subsplits_arrlst,
                         fnames_tpl=('Training_Splits.pkl', 'Testing_Splits.pkl'),
                         process_state_saving_path=None):

        # Replace the default or create if required the path where Sample Splits will be saved.
        if process_state_saving_path:
            save_path = process_state_saving_path
        else:
            save_path = self.state_save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Set the file names for training and testing splits.
        train_splits_path = save_path + fnames_tpl[0]
        test_splits_path = save_path + fnames_tpl[1]

        # Pickleing the Training and Testing Splits.
        with open(train_splits_path, 'w') as f:
            pickle.dump(train_subsplits_arrlst, f)

        with open(test_splits_path, 'w') as f:
            pickle.dump(testing_subsplits_arrlst, f)

    def LoadSplitSamples(self, fnames_tpl=('Training_Splits.pkl', 'Testing_Splits.pkl'),
                         process_state_saving_path=None):

        # Replace the path where the process-state files was supposed to be saved.
        if process_state_saving_path:
            save_path = process_state_saving_path
        else:
            save_path = self.state_save_path

        if not os.path.exists(save_path):
            raise Exception(
                "Loading Samples Splits Faild: process-state-saving-path does not exist"
            )

        # Set the file names for training and testing splits.
        train_splits_fname = save_path + fnames_tpl[0]
        test_splits_fname = save_path + fnames_tpl[1]

        if os.path.exists(train_splits_fname) and os.path.exists(test_splits_fname):

            # Unpickleing the Training and Testing Splits.
            with open(train_splits_fname, 'r') as f:
                train_subsplits_arrlst = pickle.load(f)

            with open(test_splits_fname, 'r') as f:
                testing_subsplits_arrlst = pickle.load(f)
        else:

            return None, None

        return train_subsplits_arrlst, testing_subsplits_arrlst

    def BuildCorpusMatrix(self, html_file_l, filename, tid_vocab, norm_func, encoding='utf-8'):

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

    def MaxNormalise(self, corpus_mtrx, vocab_len):

        # Getting the Maximum frequency for every document.
        max_val = np.max(corpus_mtrx.todense())

        if max_val == 0.0:

            # NOTE: Preventing division-by-zero For Documents with zero terms. This case occurs...
            # when a sub-Vocabulary is used for the experiment.
            max_val = 1.0

            # NOTE: PATCH for preventing All-Zero-Values vectors stopping the experiments.
            corpus_mtrx[:] = np.finfo(np.float32).min

        # Normalizing based on the matrix/array type.
        if ssp.issparse(corpus_mtrx):
            corpus_mtrx = ssp.csr_matrix(corpus_mtrx.todense() / max_val)
        else:
            corpus_mtrx = corpus_mtrx / max_val

        return corpus_mtrx

    def EvaluateAll(self, params_range, raw_corpus_files_path=None, encoding='utf-8'):
        """
            Parameters Template
            -------------------
            params_range = coll.OrderedDict([
               ('kfolds', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               ('train_split_step_method', [
                  (0.5, 0.1, 'rndred-trn-rest4-test'),
                  (0.5, 0.1, 'rndred-trn-fixed-test'),
               ])
               ('vocab_size', [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]),
               ('max_iter', [50, 100, 300])
               ('converg_diff', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
               ('learing_rate', [0.0003, 0.003, 0.01, 0.03, 0.1, 0.3])
            ])

        """

        # Replace the class instantiation defined variable self.corpus_files_path if any.
        if raw_corpus_files_path:
            self.corpus_files_path = raw_corpus_files_path

        if not os.path.exists(self.corpus_files_path):
            raise Exception("Corpus files path does not exist.")

        # Loading the Filename list of the corpus and their respective class tags.
        html_file_l, cls_tgs = self.LoadCrpsFnamesTags()

        # Loading the last good states list for skipping the sates already has been evaluated.
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

            # # # Create the group sequence respectively to the models parameters:
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
            splt_fname_suffix = '_'.join(
                [str(elem) for elem in params['train_split_step_method']]
            ).replace('.', '')

            if last_splt_fname_suffix != splt_fname_suffix:

                trn_fname = self.state_save_path + 'Training_Splits_' + splt_fname_suffix + '.pkl'
                test_fname = self.state_save_path + 'Testing_Splits_' + splt_fname_suffix + '.pkl'

                # Loading Training/Testing Splits.
                train_splts, test_splts = self.LoadSplitSamples((trn_fname, test_fname), '/')

                # In case 'None' has been loaded: Building and saving splits upon params for...
                # ...the next iteration will be needed.
                if not (train_splts and test_splts):

                    # Building the splits.
                    train_splts, test_splts = self.SplitSamples(
                        cls_tgs,
                        trn_percent=params['train_split_step_method'][0],
                        decrease_step=params['train_split_step_method'][1],
                        method=params['train_split_step_method'][2]
                    )

                    # Saving the splits.
                    self.SaveSplitSamples(
                        train_splts, test_splts, (trn_fname, test_fname), '/'
                    )

            # Setting initial value for the variable will be used also for not re-loading
            # ...a file has been loaded in the exact previous iteration.
            last_corpus_fname = ''

            # Running experiments for THIS params for each Sub-Split.
            for subsplt_cnt, (trn_subsplt, tst_subsplt) in enumerate(zip(train_splts, test_splts)):

                # Skipping the states that have already been tested.
                this_state_params = params.values()
                this_state_params.append(subsplt_cnt)
                # print last_goodstate_lst
                if this_state_params in last_goodstate_lst:
                    print "Skipping already tested state: ", this_state_params
                    continue

                # Appending the Group for this sub-split.
                try:
                    next_group = self.h5_res.get_node(next_group, '#'+str(subsplt_cnt))
                except:
                    next_group = self.h5_res.create_group(next_group, '#'+str(subsplt_cnt))
                save_group = next_group  # The final group results to be saved.

                # Loading corpus matrix for this Sub-Split.
                corpus_fname = self.state_save_path + 'Corpus_' +\
                    'VS' + str(params['vocab_size']) +\
                    '_Splt_' + splt_fname_suffix +\
                    '_#' + str(subsplt_cnt)

                # If not already loading the corpus matrix.
                if last_corpus_fname != corpus_fname:

                    # Loading the Corpus Matrix/Array for this Vocabulary and Sub-Split.
                    corpus_mtrx, file_obj = self.LoadCorpusMatrix(corpus_fname, '/')

                    # If 'None' corpus matrix has been loaded build it.
                    if corpus_mtrx is None:

                        vocab_fname = self.state_save_path + 'Vocab_' + 'Splt_' + splt_fname_suffix

                        # Loading the proper Vocabulary.
                        if os.path.exists(vocab_fname+'.pkl'):

                            # Loading the vocabulary.
                            print "Loading Vocabulary..."
                            with open(vocab_fname+'.pkl', 'r') as f:
                                tf_vocab = pickle.load(f)

                        else:
                            # Building the Vocabulary if not already exists.

                            print "Building Vocabulary..."

                            # Serializing the training split indeces.
                            srl_trn_spl = trn_subsplt.reshape(
                                (1, np.multiply(*trn_subsplt.shape))
                            )[0]

                            # Building the TF Vocabulary.
                            tf_vocab = self.terms_tf.build_vocabulary(
                                list(html_file_l[srl_trn_spl]),
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
                        print 'Building Corpus Matrix...'

                        corpus_mtrx, file_obj = self.BuildCorpusMatrix(
                            list(html_file_l), corpus_fname, tid_vocab,
                            norm_func=self.MaxNormalise, encoding=encoding
                        )

                        # NOTE: Saving the corpus matrix in normalized form.
                        file_obj, corpus_mtrx = self.SaveCorpusMatrix(
                            corpus_mtrx, corpus_fname, file_obj, '/'
                        )

                # Evaluating Semi-Supervised Classification Method.
                print "EVALUATING"
                clusters_y = self.semisuper_model.DoSemiSupervdClustrering(
                    trn_subsplt, tst_subsplt, corpus_mtrx, params
                )

                # Saving the assigned cluster labels for all the corpus subset under evaluation.
                self.h5_res.create_array(
                    save_group, 'clusters_y', clusters_y,
                    "The assigned cluster labels after Semi-Supervised clustering."
                )

                # Saving the set-ip hyper-parameters and convergence parameters.
                final_params = self.semisuper_model.get_params()

                # rec_type = np.dtype([('keys', 'S18'), ('values', 'float64')])

                d1_params = [
                    final_params['k_clusters'],
                    final_params['max_iter'],
                    final_params['final_iter'],
                    final_params['convg_diff'],
                    final_params['lrn_rate'],
                    final_params['ray_sigma']
                ]

                self.h5_res.create_array(
                    save_group, 'clustering_params',
                    np.array(d1_params, dtype=np.float64)
                )

                if ssp.issparse(final_params['dist_msur_params']):
                    dist_msur_params = np.diag(final_params['dist_msur_params'].toarray())
                else:
                    dist_msur_params = np.diag(final_params['dist_msur_params'])

                self.h5_res.create_array(
                    save_group, 'dist_params',
                    np.array(dist_msur_params, dtype=np.float64)
                )

                if ssp.issparse(final_params['w_violations']):
                    w_violations = final_params['w_violations'].toarray()
                else:
                    w_violations = final_params['w_violations']

                self.h5_res.create_array(
                    save_group, 'w_violations',
                    np.array(w_violations, dtype=np.float64)
                )

                # Saving the expected class labels for all the corpus subset under evaluation.

                # Serializing the training split indeces.
                srl_trn_spl = trn_subsplt.reshape((1, np.multiply(*trn_subsplt.shape)))
                srl_tst_spl = tst_subsplt.reshape((1, np.multiply(*tst_subsplt.shape)))

                # Getting the class tags for the corpus subset used for the Semi-Supervised...
                # ...Clustering Evaluation.
                subset_classtags_y = cls_tgs[
                    np.short(
                        np.hstack((srl_trn_spl, srl_tst_spl))
                    )
                ]

                self.h5_res.create_array(
                    save_group, 'expected_y', subset_classtags_y,
                    "Expected Classes per Document (CrossValidation Set)"
                )

                print

                # if model_specific_d:
                #    pass
                # for name, value in model_specific_d.items():
                # self.h5_res.create_array(kfld_group, name, value, "<Comment>")[:]

                # ONLY for PyTables Case: Safely closing the corpus matrix hd5 file.
                if file_obj:
                    file_obj.close()

                # Saving the last good state. Then the process can continue after this state in...
                # ...order not to start every Evaluation again.
                with open(self.state_save_path+'last_good_sate.jsn', 'w') as f:
                    pram_vals = params.values()
                    pram_vals.append(subsplt_cnt)
                    last_goodstate_lst.append(pram_vals)
                    json.dump(last_goodstate_lst, f)

        # Return Results H5 File handler class
        return self.h5_res


class SemiSupervisedParamGridSearchTables(SemiSupervisedParamGridSearchBase):

    def __init__(self, semisupervised_model, terms_tf_module, class_names_lst,
                 h5_file_results, raw_corpus_files_path, process_state_saving_path):

        # Passing the argument to the Super-Class
        super(SemiSupervisedParamGridSearchTables, self).__init__(
            semisupervised_model, terms_tf_module, class_names_lst,
            h5_file_results, raw_corpus_files_path, process_state_saving_path
        )

    def BuildCorpusMatrix(self, html_file_l, filename, tid_vocab, norm_func, encoding='utf-8'):

        # Setting the pyTables suffix, just for separating them from Numpy pickled Arrays/Matrices.
        filename = filename + '.h5'

        print "Building the Corpus Matrix (Tables)..."

        # Creating TF Vectors Matrix (pyTables TF EArray)
        corpus_mtrx, h5f = self.terms_tf.from_files(
            xhtml_file_l=html_file_l, h5_fname=filename, tid_vocabulary=tid_vocab,
            norm_func=norm_func, encoding=encoding, error_handling='replace'
        )[0:2]  # <--- Getting only 2 of the 3 returned values.

        # Returning the Corpus Matrix aligned upon the given Vocabulary.
        return (corpus_mtrx, h5f)

    def SaveCorpusMatrix(self, corpus_mtrx, filename, file_obj, process_state_saving_path=None):

        # Does Nothing. It is only usfull for Numpy/Scipy.sparse Arrays/Matrices.
        process_state_saving_path = None

        # Closing and re-opening file just for safety.
        file_obj.close()
        file_obj = tb.open_file(filename+'.h5', 'r+')
        corpus_mtrx = file_obj.get_node('/',  'corpus_earray')

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

            corpus_mtrx = h5f.get_node('/',  'corpus_earray')  # h5f.root.corpus_earray

        else:
            return None, None

        return (corpus_mtrx, h5f)
