

import numpy as np
import scipy.spatial.distance as spd
import sklearn.svm as svm


class OCSVMEDMPG_Wrapped(object):  # DMPG: Document Model per Genre.

    def __init__(self, genres):

        raise Exception("Needs revisiting! PLEASE DO NOT USE IT!")

        self.genres_lst = genres
        self.gnrlst_idx = dict([(g, i) for i, g in enumerate(genres)])
        self.gnrs_num = len(genres)

    def contruct_classes(self, trn_idxs, corpus_mtrx_lst, cls_gnr_tgs, params):
        inds_per_gnr = dict()
        # inds = list()
        last_gnr_tag = None

        for gnr_tag in np.unique(cls_gnr_tgs[trn_idxs]):
            inds_per_gnr[self.genres_lst[gnr_tag - 1]] = trn_idxs[
                np.where(cls_gnr_tgs[trn_idxs] == gnr_tag)[0]
            ]

        gnr_classes = dict()

        for g, inds in inds_per_gnr.items():

            # Create the OC-SVM Model for this genre
            gnr_classes[g] = svm.OneClassSVM(
                kernel='linear', nu=params['nu'], shrinking=True, cache_size=200, verbose=False
            )
            # print "Fit Model for ", g
            # print "Corpus_Mtrx", corpus_mtrx_lst[inds_per_gnr[g], :].shape
            # Convert TF vectors to Binary
            # crp_arr_bin = np.where(corpus_mtrx_lst[inds_per_gnr[g], :].toarray() > 0, 1, 0)

            # Fit OC-SVM Model to Data of this genre.
            # gnr_classes[g].fit(corpus_mtrx_lst[inds_per_gnr[g], 0::])  # params['features_size']])
            print inds_per_gnr[g], inds
            gnr_classes[g].fit(corpus_mtrx_lst[self.gnrlst_idx[g]][inds, 0::])
            # ###for Sparse Matrix it might require corpus_mtrx_lst[inds_per_gnr[g], :].toarray()

        return gnr_classes

    def predict(self, *args):

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        corpus_mtrx_lst = args[1]
        gnr_classes = args[2]

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        # Initialize Predicted-Classes-Arrays List
        predicted_Y_per_gnr = list()
        predicted_dist_per_gnr = list()

        for g in gnr_classes.keys():

            # Get the part of matrices or arrays required for the model prediction phase
            crossval_X = corpus_mtrx_lst[self.gnrlst_idx[g]][crv_idxs, 0::]
            # params['features_size']]...
            # EXTREMELY IMPORTANT: corpus_mtrx_lst[X] where X=[<idx1>,<idx2>,...,<idxN>]...
            # ...returns ERROR HDF5 when using pytables Earray. For scipy.sparse there is no...
            # ...such a problem. Therefore it always should be used this expression...
            # ...corpus_mtrx_lst[X, :]

            # Converting TF vectors to Binary
            # cv_arr_bin = np.where(crossval_X.toarray() > 0, 1, 0)

            # Getting the predictions for each Vector for this genre
            predicted_Y = gnr_classes[g].predict(crossval_X)
            # For an one-class model, +1 or -1 is returned.
            predicted_D = gnr_classes[g].decision_function(crossval_X)
            # For Sparse Matrices it might require crossval_X.toarray()

            # Assigning Genre-Class tag to Predicted_Y(s)
            predicted_Y = np.where(predicted_Y == 1, self.genres_lst.index(g) + 1, 0)

            # Keeping the prediction per genre
            predicted_Y_per_gnr.append(predicted_Y)
            predicted_dist_per_gnr.append(predicted_D.reshape(predicted_D.shape[0]))

        # Converting it to Array before returning
        predicted_Y_per_gnr = np.vstack(predicted_Y_per_gnr)
        predicted_dist_per_gnr = np.vstack(predicted_dist_per_gnr)

        # Finding index of the Max Positive distances from the Ensembles Predicted...
        # ...distance Array/Matrix
        max_dist_idxs = np.argmax(predicted_dist_per_gnr, axis=0)

        # Keeping the Max Positive distance form Predicted distances Array/Matrix and the...
        # ...respected Predicted Ys
        predicted_scores = np.choose(max_dist_idxs, predicted_dist_per_gnr)
        predicted_Y = np.choose(max_dist_idxs, predicted_Y_per_gnr)

        return (predicted_Y, predicted_scores, predicted_Y_per_gnr, predicted_dist_per_gnr)

    def eval(self, *args):

        # Get Input arguments in given sequence
        trn_idxs = args[0]
        crv_idxs = args[1]
        corpus_mtrx_lst = args[2]
        cls_gnr_tgs = args[3]
        params = args[4]

        # Build Genre Classes given the training vectors
        gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx_lst, cls_gnr_tgs, params)

        # Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
        results = self.predict(crv_idxs, corpus_mtrx_lst, params, gnr_classes)

        # Return results as expected form ParamGridCrossValBase class
        return {
            'predicted_Y': results[0],
            'predicted_scores': results[1],
            'predicted_Y_per_gnr': results[2],
            'predicted_dist_per_gnr': results[3]
        }


class OCSVME(object):

    def __init__(self, nu):
        self.nu = nu

    def contruct_classes(self, trn_idxs, corpus_mtrx, cls_gnr_tgs):

        # Getting the Unique class-tags and the Indecies for each class tag.
        inds_per_gnr = dict()
        for i, gnr_tag in enumerate(np.unique(cls_gnr_tgs[trn_idxs])):
            inds_per_gnr[i] = trn_idxs[np.where(cls_gnr_tgs[trn_idxs] == gnr_tag)[0]]

        # Training an SVM Learner for each class.
        gnr_classes = dict()
        for ctg, inds in inds_per_gnr.items():

            # Create the OC-SVM Model for this genre
            gnr_classes[ctg] = svm.OneClassSVM(
                kernel='linear', nu=self.nu, shrinking=True, cache_size=200, verbose=False
            )

            # Convert TF vectors to Binary
            # crp_arr_bin = np.where(corpus_mtrx[inds_per_gnr[g], :].toarray() > 0, 1, 0)

            # Fit OC-SVM Model to Data of this genre.
            gnr_classes[ctg].fit(corpus_mtrx[inds, 0::])
            # ###for Sparse Matrix it might require corpus_mtrx[inds_per_gnr[g], :].toarray()

        return gnr_classes

    def predict(self, *args):

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        corpus_mtrx = args[1]
        gnr_classes = args[2]

        # Get the part of matrices or arrays required for the model prediction phase
        crossval_X = corpus_mtrx[crv_idxs, 0::]

        # Initialize Predicted-Classes-Arrays List
        predicted_Y_per_gnr = list()
        predicted_dist_per_gnr = list()

        for ctg in gnr_classes.keys():

            # Getting the predictions for each Vector for this genre
            predicted_Y = gnr_classes[ctg].predict(crossval_X)
            # For an one-class model, +1 or -1 is returned.
            predicted_D = gnr_classes[ctg].decision_function(crossval_X)
            # For Sparse Matrices it might require crossval_X.toarray()

            # Assigning Genre-Class tag to Predicted_Y(s)
            predicted_Y = np.where(predicted_Y == 1, ctg, 0)

            # Keeping the prediction per genre
            predicted_Y_per_gnr.append(predicted_Y)
            predicted_dist_per_gnr.append(predicted_D.reshape(predicted_D.shape[0]))

        # Converting it to Array before returning
        predicted_Y_per_gnr = np.vstack(predicted_Y_per_gnr)
        predicted_dist_per_gnr = np.vstack(predicted_dist_per_gnr)

        # Finding index of the Max Positive distances from the Ensembles Predicted...
        # ...distance Array/Matrix
        max_dist_idxs = np.argmax(predicted_dist_per_gnr, axis=0)

        # Keeping the Max Positive distance form Predicted distances Array/Matrix and the...
        # ...respected Predicted Ys
        predicted_scores = np.choose(max_dist_idxs, predicted_dist_per_gnr)
        predicted_Y = np.choose(max_dist_idxs, predicted_Y_per_gnr)

        return (predicted_Y, predicted_scores, predicted_Y_per_gnr, predicted_dist_per_gnr)


def eval(*args):

    # Get Input arguments in given sequence
    trn_idxs = args[0]
    crv_idxs = args[1]
    corpus_mtrx = args[2]
    cls_gnr_tgs = args[3]
    params = args[4]

    print "ARG PASSED!"

    # Initilising the RFSE model.
    ocsvme = OCSVME(params['nu'])

    # Build Genre Classes given the training vectors
    gnr_classes = ocsvme.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs)

    # Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
    results = ocsvme.predict(crv_idxs, corpus_mtrx, gnr_classes)

    # Return results as expected form ParamGridCrossValBase class
    return {
        'predicted_Y': results[0],
        'predicted_scores': results[1],
        'predicted_Y_per_gnr': results[2],
        'predicted_dist_per_gnr': results[3]
    }
