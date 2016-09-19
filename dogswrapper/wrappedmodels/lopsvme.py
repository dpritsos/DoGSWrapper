

import numpy as np
import scipy.spatial.distance as spd
import svmrop


class LOPSVM_Wrapped(object):

    def __init__(self, genres):
        self.genres_lst = genres
        self.gnrs_num = len(genres)

    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, params):
        inds_per_gnr = dict()
        inds = list()
        last_gnr_tag = 1

        for trn_idx in trn_idxs:

            if cls_gnr_tgs[trn_idx] != last_gnr_tag:
                inds_per_gnr[self.genres_lst[last_gnr_tag - 1]] = inds
                last_gnr_tag = cls_gnr_tgs[trn_idx]
                inds = []

            inds.append(trn_idx)
        print self.genres_lst
        inds_per_gnr[self.genres_lst[last_gnr_tag - 1]] = inds

        gnr_classes = dict()
        for g in self.genres_lst:

            # Create the OC-SVM Model for this genre
            if params['svm_type'] == 'oneclass':
                self.lopsvm = svmrop.LinearSetSVM(
                    svm_type='oneclass',
                    l=params['l'],
                    c1_w=params['c1_w'], c2_w=params['c2_w'],
                    mrgn_nw=params['mrgn_nw'], mrgn_fw=params['mrgn_fw'],
                    nu=params['nu']
                )
            else:
                self.lopsvm = svmrop.LinearSetSVM(
                    svm_type='binary',
                    l=params['l'],
                    c1_w=params['c1_w'], c2_w=params['c2_w'],
                    mrgn_nw=params['mrgn_nw'], mrgn_fw=params['mrgn_fw'],
                    penalty='l2', multi_class='ovr', dual=True
                )
            # print "Fit Model for ", g
            # print "Corpus_Mtrx", corpus_mtrx[inds_per_gnr[g], :].shape
            # Convert TF vectors to Binary
            # crp_arr_bin = np.where(corpus_mtrx[inds_per_gnr[g], :].toarray() > 0, 1, 0)

            # ### Fitting Linear 1-vs-set SVM Model to Data of this genre. ###

            # Getting the positive samples for this split
            yp_i = inds_per_gnr[g]

            # Getting some negative samples for this split.
            yn_i = np.hstack([np.random.shuffle(inds_per_gnr[g])[0:5] for g in self.genres_lst])

            print yn_i

            0/0

            # Not in use yet!
            yu_i = 0

            y = np.hstack((yp_i, yn_i))

            near_H, far_H = self.lopsvm.optimize(
                corpus_mtrx[y, 0:params['features_size']], yp_i, yn_i, yu_i
            )

            near_H, far_H = self.lopsvm.refine_planes(yp_i, yn_i)

            gnr_classes[g] = [near_H, far_H]

        return gnr_classes

    def predict(self, *args):

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        corpus_mtrx = args[1]
        params = args[2]
        gnr_classes = args[3]

        # Get the part of matrices or arrays required for the model prediction phase
        crossval_X = corpus_mtrx[crv_idxs, 0:params['features_size']]
        # EXTREMELY IMPORTANT: corpus_mtrx[X] where X=[<idx1>,<idx2>,...,<idxN>] returns ERROR...
        # ...HDF5 when using pytables Earray. For scipy.sparse there is no such a problem....
        # Therefore it always should be used this expression corpus_mtrx[X, :]

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        # Initialize Predicted-Classes-Arrays List
        predicted_Y_per_gnr = list()
        predicted_dist_per_gnr = list()

        for g in self.genres_lst:

            # Converting TF vectors to Binary
            # cv_arr_bin = np.where(crossval_X.toarray() > 0, 1, 0)

            # Getting the predictions for each Vector for this genre
            predicted_Y, predicted_D = self.lopsvm.predict(
                crossval_X, gnr_classes[g][0], gnr_classes[g][1]
            )

            # Keeping the prediction per genre
            predicted_Y_per_gnr.append(predicted_Y)
            predicted_dist_per_gnr.append(predicted_D)

        # Converting it to Array before returning
        predicted_Y_per_gnr = np.vstack(predicted_Y_per_gnr)
        predicted_dist_per_gnr = np.vstack(predicted_dist_per_gnr)

        return predicted_Y_per_gnr, predicted_dist_per_gnr

    def eval(self, *args):

        # Get Input arguments in given sequence
        trn_idxs = args[0]
        crv_idxs = args[1]
        corpus_mtrx = args[2]
        cls_gnr_tgs = args[3]
        vocab_index_dct = args[4]  # tid
        params = args[5]

        # Build Genre Classes given the training vectors
        gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs, params)

        # Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
        results = self.predict(crv_idxs, corpus_mtrx, params, gnr_classes)

        # Expected Results for the ParamGridCrossValBase class in paramgridcrossval module
        predicted_Y = results[0]
        predicted_scores = results[1]
        model_specific_d = None

        # Return results as expected form ParamGridCrossValBase class
        return predicted_Y, predicted_scores
