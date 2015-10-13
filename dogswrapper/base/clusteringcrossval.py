
import sys
sys.path.append('../../html2vectors/src')

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
from html2vect.utils import tfdutils
from html2vect.base.io.basefilehandlers import file_list_frmpaths


class SemiSupervisedParamGridSearchBase(object):

    def __init__(self, semisupervised_model, terms_tf_module, h5_file_results, class_names_lst):
        self.semisuper_model = terms_type_tf_module
        self.terms_tf = terms_tf_module
        self.classes_lst = class_names_lst
        self.classes_num = len(class_names_lst)
        self.h5_res = h5_file_results

    def LoadCrpsFnamesTags(self, raw_corpus_files_path, saving_path):

        corpus_files_lst_path = saving_path+'/Corpus_filename_shorted.lst'
        corpus_tags_lst_path = saving_path+'/Corpus_tags_shorted.lst'

        if os.path.exists(corpus_files_lst_path) and os.path.exists(corpus_files_lst_path):

            print "Loading HTML Filenames list and Classes Tags Numpy array..."

            # Load Filename and classes Tags lists
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
                gnrs_file_lst = file_list_frmpaths(self.corpus_path, [str(g + "/html/")])

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
        return html_file_l, np.array(cls_tgs)

    def SplitSample(self, cls_tgs, trn_percent=0.5, decrease_step=0.1, method='on-file-prefix'):

        # Checking trn_percent and decrease_step value constraints.
        if trn_percent < 0.01 or trn_percent > 1.0 or decrease_step < 0.01 or decrease_step > 1.0:
            raise Exception("trm_percent and decrease_step values mast be in range [0.01, 1]")

        # Selecting the method to split the corpus to training and test sets.
        if method == 'on-file-prefix':

            # Two list of arrays where each array has the file indeces for training and testing...
            # ...repspectivly splitted initially upon trn_percentage.
            trn_splts_per_ctg_arrlst, trn_splts_per_ctg_arrlst = list(), list()

            for ctg in np.unique(cls_tgs):

                # Getting the filename list indeces for this class (tag).
                this_cls_idxs = np.where(cls_tgs == ctg)[0]

                # Calculating the amount of samples keeping for training for this class for the...
                # ...initial split.
                smpls_num = np.ceil(this_cls_idxs.shape[0] * trn_percent)

                # NOTE: Here the training indeces are selected Randomly! Thus, it is easy to...
                # ...use this python-class-method into Cross-Validation experimental set-up....
                # ...The list of indeces is sorted.
                train_files_idxs_arr = np.sort(
                    np.random.choice(this_cls_idxs, smpls_num, replace=False)
                )

                trn_splts_per_ctg_arrlst.append(train_files_idxs_arr)

                # Keeping the indeces but the ones for training as testing indeces.
                trn_splts_per_ctg_arrlst.append(
                    np.array(
                        [tst_i for tst_i in this_cls_idxs if tst_i not in train_files_idxs_arr]
                    )
                )

            # Two lists per sub-split one for training and one for testing. Every element of the...
            # ...list is containing an array where the rows are containing the training and...
            # ...testing index splits for every class (tag) respectively.
            train_subsplits_arrlst, testing_subsplits_arrlst = list(), list()

            for splts_decreased_perc in np.arange(1, decrease_step, -decrease_step):

                train_ctg_lst, test_ctg_lst = list(), list()

                for trn_splts_per_ctg_arr in trn_splts_per_ctg_arrlst:

                    smpls_num = np.ceil(trn_splts_per_ctg_arr.shape[0] * splts_decreased_perc)

                    train_ctg_lst.append(trn_splts_per_ctg_arr[0:smpls_num])
                    test_ctg_lst.append(trn_splts_per_ctg_arrlst)

                # Keeping the sub-splits array lists.
                train_subsplits_arrlst.append(
                    np.vstack(train_ctg_lst)
                )

                testing_subsplits_arrlst.append(
                    np.vstack(test_ctg_lst)
                )

        else:
            raise Exception("Non-implemented yet!")

        return train_subsplits_arrlst, testing_subsplits_arrlst

    def BuildVocabularyOnParams(self, html_file_l, train_idxs, params):
        pass

    def LoadVocabularyOnParams(self, train_idxs, params):
        pass

    def BuildCorpusMatrixOnParams(self, html_file_l, tid_vocab, params):
        pass

    def LoadCorpusMatrixONParams(self, params):

        # # # # params==(k, vocab_size_group, vocab_size, html_file_l, tid, norm_func)

        # Load or Create the Corpus Matrix (Spase) for this combination or k-fold and vocab_size
        corpus_mtrx_fname = self.crps_voc_path+'/kfold_CorpusMatrix_'+str(k)+str(vocab_size)+'.pkl'

        if os.path.exists(corpus_mtrx_fname):

            print "Loading Sparse TF Matrix for CrossValidation for K-fold=", k,
            " and Vocabulary size=", vocab_size

            # Loading Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
            with open(corpus_mtrx_fname, 'r') as f:
                corpus_mtrx = pickle.load(f)

        else:

            print "Creating Sparse TF Matrix (for CrossValidation) for K-fold=", k,
            " and Vocabulary size=", vocab_size

            # Creating TF Vectors Sparse Matrix
            corpus_mtrx = self.terms_tf.from_files(
                list(html_file_l), tid_vocabulary=tid, norm_func=norm_func,
                encoding='utf8', error_handling='replace'
            )[0]  # <--- Be careful with zero index

            # Save the Web-pages term counts (Char N-grans or Word N-Grams)
            kfld_group = self.h5_res.create_group(vocab_size_group, 'KFold'+str(k))
            docs_term_counts = self.h5_res.create_array(
                kfld_group, 'docs_term_counts', np.sum(corpus_mtrx.toarray(), axis=1)
            )

            # Perform default (division by max value) normalization for corpus matrix 'corpus_mtrx'
            # Should I perform Standarisation/Normalisation by subtracting mean value from...
            # ...vector variables?
            print "Normalizing"

            # Getting the Maximum frequency for every document.
            max_vals = np.max(corpus_mtrx.todense(), axis=1)

            # For Documents with zero terms. This case occurs when a sub-Vocabulary is used for...
            # ...the experiment.
            max_vals[np.where(max_vals == 0)] = 1

            # Normalizing.
            corpus_mtrx = ssp.csr_matrix(corpus_mtrx.todense() / max_vals)

            # Saving TF Vectors Matrix
            print "Saving Sparse Normalized TF Matrix (for CrossValidation)"
            with open(corpus_mtrx_fname, 'w') as f:
                pickle.dump(corpus_mtrx, f)

        return (corpus_mtrx, None)

    def EvaluateSemiSupervisedModel(self, *args):
        """ Call prototyping: evaluate(html_file_l, cls_tgs, None, params_range, 'utf-8') """

        html_file_l = args[0]
        cls_tgs = args[1]
        test_only_tgs = args[2]
        norm_func = args[3]
        params_range = args[4]
        encoding = args[5]

        """
        # Create CrossVal Folds
        KF = cross_validation.StratifiedKFold(
             cls_tgs, len(params_range['kfolds']), indices=True
        )

        for k, (trn, crv) in enumerate(KF):

            voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.vtf'
            pkl_voc_filename = self.crps_voc_path+'/kfold_Voc_'+str(k)+'.pkl'
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'

            # Save K-Fold Cross-Validation corpus vector selection-indecies if does not exists...
            # ...Create K-fold Cross-Validation Vocabulary for each fold Stratified Indeces and...
            # ...respective Vocabularies should be synchronized therefore there saving-files...
            # ...should be created all together if one or more are missing.
            if not os.path.exists(trn_filename) or not os.path.exists(crv_filename) or\
                    not os.path.exists(voc_filename) or not os.path.exists(pkl_voc_filename):

                # Save Training Indices
                print "Saving Training Indices for k-fold=", k
                with open(trn_filename, 'w') as f:
                    json.dump(list(trn), f, encoding=encoding)

                # Save Cross-validation Indices
                print "Saving Cross-validation Indices for k-fold=", k
                with open(crv_filename, 'w') as f:
                    if test_only_tgs:
                        json.dump(list(crv) + test_only_idxs, f, encoding=encoding)
                    else:
                        json.dump(list(crv), f, encoding=encoding)

                # Creating Vocabulary
                print "Creating Vocabulary for k-fold=", k
                tf_d = self.terms_tf.build_vocabulary(
                    list(html_file_l[trn]), encoding='utf-8', error_handling='replace'
                )

                # Saving Vocabulary
                print "Saving Vocabulary"
                with open(pkl_voc_filename, 'w') as f:
                    pickle.dump(tf_d, f)

                with open(voc_filename, 'w') as f:
                    json.dump(tf_d, f, encoding=encoding)
        """

        # Starting Parameters Grid Search
        for gci, params in enumerate(param_combs.ParamGridIter(params_range)):

            # Show how many Grid Search Parameter combinations are renaming.
            print "Param Grid Counts:", gci+1

            # Prevent execution of this loop in case feature_size is smaller than Vocabulary size.
            if params['features_size'] > params['vocab_size']:
                print "SKIPPED Params: ", params
                continue

            print "Params: ", params
            # bs = cross_validation.Bootstrap(9, random_state=0)
            # Set Experiment Parameters
            k = params['kfolds']
            vocab_size = params['vocab_size']
            featrs_size = params['features_size']

            # Creating a Group for this Vocabulary size in h5 file under this k-fold
            try:
                vocab_size_group = self.h5_res.get_node('/', 'vocab_size'+str(vocab_size))
            except:
                vocab_size_group = self.h5_res.create_group(
                    '/', 'vocab_size'+str(vocab_size),
                    "Vocabulary actual size group of Results Arrays for this K-fold"
                )

            # Creating a Group for this features size in h5 file under this k-fold
            try:
                feat_num_group = self.h5_res.get_node(
                    vocab_size_group, 'features_size'+str(featrs_size)
                )
            except:
                feat_num_group = self.h5_res.create_group(
                    vocab_size_group, 'features_size'+str(featrs_size),
                    "Features Number group of Results Arrays for this K-fold"
                )

            # # # Create the group sequence respectively to the models parameters:

            # Assigning Feature number group to next_group parameter for initializing the loop
            next_group = feat_num_group

            # Start the loop of creating or getting group nodes in respect to model parameters
            for pname, pvalue in params.items():
                if pname not in ['kfolds', 'vocab_size', 'features_size']:
                    try:
                        next_group = self.h5_res.get_node(
                            next_group, pname+str(pvalue).replace('.', '')
                        )
                    except:
                        next_group = self.h5_res.create_group(
                            next_group, pname+str(pvalue).replace('.', ''), "<Comment>"
                        )

            # # # END- Group creation sequence

            # Creating a Group for this k-fold in h5 file
            try:
                kfld_group = self.h5_res.get_node(next_group, 'KFold'+str(k))
            except:
                kfld_group = self.h5_res.create_group(
                    next_group, 'KFold'+str(k), "K-Fold group of Results Arrays"
                )

            # Loading Vocabulary
            print "Loading VOCABULARY for k-fold=", k
            with open(pkl_voc_filename, 'r') as f:
                tf_d = pickle.load(f)

            # Get the Vocabulary keeping all the terms with same freq to the last feature of...
            # ...the requested size.
            resized_tf_d = tfdutils.keep_atleast(tf_d, vocab_size)

            # Saving the real Vocabulary sizes for this experiment (i.e. this text...
            # ...representation, etc.) keep it as pytables group attribute the actual...
            # ...Vocabulary size
            vocab_size_group._v_attrs.real_voc_size = [(k, len(resized_tf_d))]

            # Create The Terms-Index Vocabulary that is shorted by Frequency descending order
            tid = tfdutils.tf2tidx(resized_tf_d)
            # print tid.items()[0:5]

            # Load or Create the Coprus Matrix/Array for this combination or kfold and vocab_size.
            corpus_mtrx, corpus_file = self.corpus_matrix(
                k, vocab_size_group, vocab_size, html_file_l, tid, norm_func
            )

            # Load Training Indeces
            trn_filename = self.crps_voc_path+'/kfold_trn_'+str(k)+'.idx'
            print "Loading Training Indices for k-fold=", k
            with open(trn_filename, 'r') as f:
                trn_idxs = np.array(json.load(f, encoding=encoding))

            # Load Cross-validation Indeces
            crv_filename = self.crps_voc_path+'/kfold_crv_'+str(k)+'.idx'
            print "Loading Cross-validation Indices for k-fold=", k
            with open(crv_filename, 'r') as f:
                crv_idxs = np.array(json.load(f, encoding=encoding))

            print "EVALUATE"
            # Evaluating Classification Method
            predicted_Y, predicted_scores, model_specific_d = self.model.eval(
                trn_idxs,
                crv_idxs,
                corpus_mtrx,
                cls_tgs,
                tid, params
            )

            # Select Cross Validation Set
            # print cls_tgs
            # print crv_idxs
            crossval_Y = cls_tgs[crv_idxs]

            P_per_gnr, R_per_gnr, F1_per_gnr = self.calculate_p_r_f1(crossval_Y, predicted_Y)

            # Saving results
            self.h5_res.create_array(
                kfld_group, 'expected_Y', crossval_Y,
                "Expected Classes per Document (CrossValidation Set)"
            )[:]
            self.h5_res.create_array(
                kfld_group, 'predicted_Y', predicted_Y,
                "predicted Classes per Document (CrossValidation Set)"
            )[:]
            self.h5_res.create_array(
                kfld_group, 'predicted_scores', predicted_scores,
                "predicted Scores per Document (CrossValidation Set)"
            )[:]
            self.h5_res.create_array(
                kfld_group, "P_per_gnr", P_per_gnr, "Precision per Genre (P[0]==Global P)"
            )[:]
            self.h5_res.create_array(
                kfld_group, "R_per_gnr", R_per_gnr, "Recall per Genre (R[0]==Global R)"
            )[:]
            self.h5_res.create_array(
                kfld_group, "F1_per_gnr", F1_per_gnr, "F1_statistic per Genre (F1[0]==Global F1)"
            )[:]

            print

            if model_specific_d:
                for name, value in model_specific_d.items():
                    self.h5_res.create_array(kfld_group, name, value, "<Comment>")[:]

            # Closing corpus file if any. Originally for closing hd5 files
            if corpus_file:
                corpus_file.close()

        # Return Results H5 File handler class
        return self.h5_res


class ParamGridCrossValTables(ParamGridCrossValBase):

    def __init__(self, ML_Model, terms_tf, h5_res, genres, corpus_path, voc_path):

        # Passing the argument to the Super-Class
        super(ParamGridCrossValTables, self).__init__(
            ML_Model, terms_tf, h5_res, genres, corpus_path, voc_path
        )

    def corpus_matrix(self, k, vocab_size_group, vocab_size, html_file_l, tid, norm_func):

        # Load or Create the Corpus Matrix (Spase) for this combination or kfold and vocabulary_size
        corpus_mtrx_fname = self.crps_voc_path+'/kfold_CorpusMatrix_'+str(k)+str(vocab_size)+'.h5'

        if os.path.exists(corpus_mtrx_fname):

            print "Loading pyTables TF EArray for CrossValidation for K-fold=", k,
            " and Vocabulary size=", vocab_size
            # Loading Corpus pyTables TF EArray for this combination or kfold and vocabulary_size
            h5f = tb.open_file(corpus_mtrx_fname, 'r+')
            corpus_mtrx = h5f.get_node('/',  'corpus_earray')  # h5f.root.corpus_earray

        else:

            print "Creating pyTables TF EArray (for CrossValidation) for K-fold=", k,
            " and Vocabulary size=", vocab_size

            # Creating pyTables TF EArray.
            corpus_mtrx, h5f = self.terms_tf.from_files(
                list(html_file_l), corpus_mtrx_fname, tid_vocabulary=tid, norm_func=norm_func,
                encoding='utf8', error_handling='replace'
            )[0:2]  # <--- Getting only 2 of the 3 returned values.

            # Save the Webpages term counts (Char N-grans or Word N-Grams).
            kfld_group = self.h5_res.create_group(vocab_size_group, 'KFold'+str(k))
            docs_term_counts = self.h5_res.create_array(
                kfld_group, 'docs_term_counts', np.sum(corpus_mtrx, axis=1)
            )

            # Performing default (division by max value) normalization for corpus matrix...
            # ...'corpus_mtrx'. Should I perform Standarisation/Normalisation by subtracting mean...
            # ...value from vector variables? The following way of normalization supposed to be...
            # ...executed faster and it requires much less memory because it prevents the use of...
            # ...indermidate array which it is required in 'c = c / c.max()' operation.
            print "Normalizing"
            max_col_arr = np.max(corpus_mtrx, axis=1)[:, np.newaxis]

            # For Documents with zero terms. This case occurs when a sub-Vocabulary is used for...
            # ...the experiment.
            max_col_arr[np.where(max_col_arr == 0)] = 1

            for i, (row, max_val) in enumerate(zip(corpus_mtrx.iterrows(), max_col_arr)):
                corpus_mtrx[i] = row / max_val

        return (corpus_mtrx, h5f)
