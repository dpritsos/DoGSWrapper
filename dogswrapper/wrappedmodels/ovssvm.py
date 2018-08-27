
import numpy as np
import scipy.spatial.distance as spd
import subprocess as subp


class OVSSVM_Wrapped(object):

    def __init__(self, fpath, genres):
        self.fpath = fpath
        self.genres_lst = genres
        self.gnrs_num = len(genres)

    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, params):

        print "TRAINING"

        inds_per_gnr = dict()
        inds = list()
        last_gnr_tag = 1

        for gnr_tag in np.unique(cls_gnr_tgs[trn_idxs]):
            inds_per_gnr[self.genres_lst[gnr_tag - 1]] = trn_idxs[
                np.where(cls_gnr_tgs[trn_idxs] == gnr_tag)[0]
            ]

        gnr_classes = dict()
        for g, inds in inds_per_gnr.items():

            # Create the OC-SVM Model for this genre
            if params['svm_type'] == 'oneclass':

                mrgn = str(params['mrgn_nw']) + '_' + str(params['mrgn_fw'])
                ovssvm_fname = self.fpath + 'ovssvm_' + mrgn + '_' + '.model'
                running_cmd = [
                    self.fpath + 'svm-train', '-s', '5', '-t', '0', '-G ',
                    str(params['mrgn_nw']), str(params['mrgn_fw'])
                ]

            else:

                mrgn = str(params['mrgn_nw']) + '_' + str(params['mrgn_fw'])
                ovssvm_fname = self.fpath + 'ovssvm_' + mrgn + '_' + '.model'
                running_cmd = [
                    self.fpath + 'svm-train', '-s', '7', '-t', '1', '-b', '0.5', '-G ',
                    str(params['mrgn_nw']), str(params['mrgn_fw']),
                    '-N'
                ]

            # ### Fitting Linear 1-vs-set SVM Model to Data of this genre. ###

            # Getting the positive samples for this split
            yp_i = inds

            # Getting some negative samples for this split.
            # print inds_per_gnr.items()
            # print [inds for grn, inds in inds_per_gnr.items() if grn != g]
            yn_i = np.hstack(
                [
                    np.random.permutation(inds)[0:int(np.floor(inds.shape[0]/2.0))]
                    for grn, inds in inds_per_gnr.items() if grn != g
                ]
            )

            # Not in use yet!
            yu_i = 0

            # Preparing the input file.
            train_fname = self.fpath + 'training_' + mrgn + '.dat'
            with open(train_fname, 'w') as f:

                # Creating the pisitive samples recoreds.
                for X in corpus_mtrx[yp_i, 0:params['features_size']]:
                    f.write('1 ' + ' '.join([str(i)+':'+str(x) for i, x in enumerate(X)]) + '\n')

                # Creating the negative samples recoreds.
                for x in corpus_mtrx[yn_i, 0:params['features_size']]:
                    f.write('-1 ' + ' '.join([str(i)+':'+str(x) for i, x in enumerate(X)]) + '\n')

            # Training the SVM 1-vs-set
            running_cmd.extend([train_fname, ovssvm_fname])
            print subp.call(running_cmd)

            # Keeping the Decision function, i.e. the decision hyperplanes for the evaluation.
            with open(ovssvm_fname, 'r') as f:

                for ln in f:

                    ln_cont_lst = ln.split(' ')

                    if ln_cont_lst[0] == 'alpha':
                        near_H = float(ln_cont_lst[-1][0:-1])

                    if ln_cont_lst[0] == 'omega':
                        far_H = float(ln_cont_lst[-1][0:-1])

                gnr_classes[g] = [near_H, far_H, ovssvm_fname]

        return gnr_classes
    """
    def predict(self, X, near_H, far_H):

        dsz_N, dsz_F = self.dfunc(X, near_H, far_H)

        return np.where(((dsz_N >= 0) & (dsz_F >= 0)), 1, -1), dsz_N, dsz_F

    def predictions(self, X, near_H, far_H):

        dsz_N, dsz_F = self.dfunc(X, near_H, far_H)

        return np.where(((dsz_N >= 0) & (dsz_F >= 0)), 1, -1)

    def dfunc(self, X, near_H, far_H):

        dsz_X = self.lsvm.decision_function(X)
        if self.svm_type == 'oneclass':
            dsz_X = np.transpose(dsz_X)[0]

        dsz_N = far_H - dsz_X
        dsz_F = dsz_X - near_H

        return dsz_N, dsz_F
    """
    def predict(self, *args):

        print 'PREDICTING'

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        expected_Y = args[1]
        corpus_mtrx = args[2]
        params = args[3]
        gnr_classes = args[4]

        # Get the part of matrices or arrays required for the model prediction phase
        crossval_X = corpus_mtrx[crv_idxs, 0:params['features_size']]
        # EXTREMELY IMPORTANT: corpus_mtrx[X] where X=[<idx1>,<idx2>,...,<idxN>] returns ERROR...
        # ...HDF5 when using pytables Earray. For scipy.sparse there is no such a problem....
        # Therefore it always should be used this expression corpus_mtrx[X, :]

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        # Initialize Predicted-Classes-Arrays List
        predicted_Y_per_gnr = list()
        predicted_d_near_per_gnr = list()
        predicted_d_far_per_gnr = list()
        gnr_cls_idx = list()

        for g in gnr_classes.keys():

            # Converting TF vectors to Binary
            # cv_arr_bin = np.where(crossval_X.toarray() > 0, 1, 0)

            # Preparing the input file for preditions
            mrgn = str(params['mrgn_nw']) + '_' + str(params['mrgn_fw'])
            test_fname = self.fpath + 'testing_' + mrgn + '.dat'
            with open(test_fname, 'w') as f:

                # Creating the positive and negative tagggind expected for this algorithm.
                yp_i = crv_idxs[np.where((expected_Y == self.genres_lst.index(g) + 1))[0]]
                yn_i = crv_idxs[np.where((expected_Y != self.genres_lst.index(g) + 1))[0]]

                # Creating the pisitive samples recoreds.
                # print 'POS', yp_i, params['features_size']
                for X in corpus_mtrx[yp_i, 0:params['features_size']]:
                    f.write('1 ' + ' '.join([str(i)+':'+str(x) for i, x in enumerate(X)]) + '\n')

                # Creating the negative samples recoreds.
                # print 'NEG', yn_i, params['features_size']
                for x in corpus_mtrx[yn_i, 0:params['features_size']]:
                    f.write('-1 ' + ' '.join([str(i)+':'+str(x) for i, x in enumerate(X)]) + '\n')

            # Getting the predictions for each Vector for this genre
            outputfile = self.fpath + 'output_' + mrgn + '.dat'
            ovssvm_fname = gnr_classes[g][2]
            running_cmd = [
                self.fpath + 'svm-predict', '-o', '-a', test_fname, ovssvm_fname, outputfile
            ]
            print subp.call(running_cmd)
            # predicted_Y, predicted_D_near, predicted_D_far =
            # svmrop.LinearSetSVM.__class__.predict(
            #    crossval_X, gnr_classes[g][0], gnr_classes[g][1]
            # )

            # Keeping the prediction per genre
            with open(outputfile, 'r') as f:

                # Creating the array for keeping the predictions.
                predicted_Y = np.zeros(crv_idxs.shape[0], dtype=np.float)

                # Reading the predicitons.
                c = 0
                c2 = 0
                for i, ln in enumerate(f):

                    predicted_Y[i] = float(ln[0:2])
                    if predicted_Y[i] > 0:
                        c += 1

                    if predicted_Y[i] < 0:
                        c2 += 1

                print i, c, c2

                predicted_Y_per_gnr.append(predicted_Y)
                # TEST TEST
                predicted_d_near_per_gnr.append(predicted_Y)
                predicted_d_far_per_gnr.append(predicted_Y)
                # TEST TEST
                gnr_cls_idx.append(self.genres_lst.index(g) + 1)

        # Converting it to Array before returning
        predicted_Y_per_gnr = np.vstack(predicted_Y_per_gnr)
        predicted_d_near_per_gnr = np.vstack(predicted_d_near_per_gnr)
        predicted_d_far_per_gnr = np.vstack(predicted_d_far_per_gnr)
        gnr_cls_idx = np.hstack(gnr_cls_idx)

        return predicted_Y_per_gnr, predicted_d_near_per_gnr, predicted_d_far_per_gnr, gnr_cls_idx


def eval(self, *args):

    # Get Input arguments in given sequence
    trn_idxs = args[0]
    crv_idxs = args[1]
    expected_Y = args[2]
    corpus_mtrx = args[3]
    cls_gnr_tgs = args[4]
    params = args[5]

    # Build Genre Classes given the training vectors
    gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs, params)

    # Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
    results = self.predict(crv_idxs, expected_Y, corpus_mtrx, params, gnr_classes)

    # Expected Results for the ParamGridCrossValBase class in paramgridcrossval module
    predicted_Y = results[0]
    predicted_d_near_per_gnr = results[1]
    predicted_d_far_per_gnr = results[2]
    gnr_cls_idx = results[3]

    # Return results as expected form ParamGridCrossValBase class
    return predicted_Y, predicted_d_near_per_gnr, predicted_d_far_per_gnr, gnr_cls_idx
