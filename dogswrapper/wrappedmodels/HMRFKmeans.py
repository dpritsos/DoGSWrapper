

import numpy as np
import scipy.spatial.distance as spd


class HMRFKmeans_Wrapped(object):

    def __init__(self, class_lst):
        self.class_lst = class_lst
        self.gnrs_num = len(class_lst)

    def BuildContraints(self, train_ctg_arr):

        # Building must-link constraints.
        must_lnk = list()

        for train_ctg_idxs_row in train_ctg_arr:

            for i, idx1 in enumerate(train_ctg_idxs_row[0:-1]):

                for idx2 in train_ctg_idxs_row[i+1::]:

                    must_lnk.append(set([idx1, idx2]))

        # Building cannot-link constraints.
        cannot_lnk = list()
        for i, train_ctg_idxs_row in enumerate(train_ctg_arr[0:-1, :]):

            for train_ctg_idxs_row_next in train_ctg_arr[i+1::, :]:

                cannot_lnk.extend([
                    set([idx1, idx2]) for idx1 in train_ctg_idxs_row
                    for idx2 in train_ctg_idxs_row_next
                ])

        return must_lnk, cannot_lnk

    def predict(self, *args):

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        corpus_mtrx = args[1]
        cls_gnr_tgs = args[2]
        vocab_index_dct = args[3]  # tid
        params = args[4]

        # Store the argument 5 (6th) to the proper variable
        if self.bagging and isinstance(args[5], np.ndarray):
            trn_idxs = args[5]

        elif not self.bagging and isinstance(args[5], dict):
            gnr_classes = args[5]

        else:
            raise Exception(
                'predict(): Invalid Argument, either bagging trigged with not train-index' +
                'array or non-bagging with not genre-classes argument'
            )

        # Get the part of matrices or arrays required for the model prediction phase.
        crossval_X = corpus_mtrx[crv_idxs, :]
        # NOTE: EXTREMELY IMPORTANT! corpus_mtrx[X] where X=[<idx1>,<idx2>,...,<idxN>] returns...
        # ...ERROR HDF5 when using pytables Earray.  For scipy.sparse there is no such a...
        # ...problem. Therefore it always should be used this expression corpus_mtrx[X, :]

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        max_sim_scores_per_iter = np.zeros((params['Iterations'], crossval_X.shape[0]))
        predicted_classes_per_iter = np.zeros((params['Iterations'], crossval_X.shape[0]))

        # Measure similarity for i iterations i.e. for i different feature subspaces Randomly...
        # ...selected
        for I in range(params['Iterations']):

            # print "Construct classes"
            # Construct Genres Class Vectors form Training Set. In case self.bagging is True.
            if self.bagging:
                gnr_classes = self.contruct_classes(
                    trn_idxs, corpus_mtrx, cls_gnr_tgs, params['Bagging']
                )

            # Randomly select some of the available features
            shuffled_vocabilary_idxs = np.random.permutation(np.array(vocab_index_dct.values()))
            features_subspace = shuffled_vocabilary_idxs[0: params['features_size']]

            # Initialized Predicted Classes and Maximum Similarity Scores Array for this i iteration
            predicted_classes = np.zeros(crossval_X.shape[0])
            max_sim_scores = np.zeros(crossval_X.shape[0])

            # Measure similarity for each Cross-Validation-Set vector to each available Genre...
            # ...Class(i.e. Class-Vector). For This feature_subspace.
            for i_vect, vect in enumerate(crossval_X[:, features_subspace]):

                # Convert TF vectors to Binary
                # vect_bin = np.where(vect[:, :].toarray() > 0, 1, 0)
                # NOTE: with np.where Always use A[:] > x instead of A > x in case of...
                # ...Sparse Matrices
                # print vect.shape

                max_sim = self.sim_min_value
                for cls_tag, g in enumerate(self.class_lst):

                    # Convert TF vectors to Binary
                    # gnr_cls_bin = np.where(gnr_classes[g][:, features_subspace] > 0, 1, 0)
                    # print gnr_cls_bin.shape

                    # Measure Similarity
                    if gnr_classes[g].ndim == 2:
                        # This case is called when a Sparse Matrix is used which is alway 2D...
                        # ...with first dim == 1
                        sim_score = self.similarity_func(vect, gnr_classes[g][:, features_subspace])

                    elif gnr_classes[g].ndim == 1:
                        # This case is called when a Array or pyTables-Array is used which it...
                        # ...this case should be 1D
                        sim_score = self.similarity_func(vect, gnr_classes[g][features_subspace])

                    else:
                        raise Exception(
                            "Unexpected Centroid Vector Dimensions: its shape should be " +
                            "(x,) for 1D array or (1,x) for 2D array or matrix"
                        )

                    # Just for debugging for
                    # if sim_score < 0.0:
                    #     print "ERROR: Similarity score unexpected value ", sim_score

                    # Assign the class tag this vector is most similar and keep the respective...
                    # ...similarity score.
                    if sim_score > max_sim:
                        predicted_classes[i_vect] = cls_tag + 1
                        # ###plus 1 is the real class tag 0 means uncategorized.
                        max_sim_scores[i_vect] = sim_score
                        max_sim = sim_score

            # Store Predicted Classes and Scores for this i iteration
            max_sim_scores_per_iter[I, :] = max_sim_scores[:]
            predicted_classes_per_iter[I, :] = predicted_classes[:]

        predicted_Y = np.zeros((crossval_X.shape[0]), dtype=np.float)
        predicted_scores = np.zeros((crossval_X.shape[0]), dtype=np.float)

        for i_prd_cls, prd_cls in enumerate(predicted_classes_per_iter.transpose()):
            class_occs = np.histogram(prd_cls.astype(np.int), bins=np.arange(self.gnrs_num+2))[0]
            # NOTE: One Bin per Genre plus one i.e the first to be always zero
            # print class_occs
            class_probs = class_occs.astype(np.float) / np.float(params['Iterations'])
            # print class_probs
            if np.max(class_probs) >= params['Sigma']:
                predicted_Y[i_prd_cls] = np.argmax(class_probs)
                predicted_scores[i_prd_cls] = np.max(class_probs)

        return predicted_Y, predicted_scores, max_sim_scores_per_iter, predicted_classes_per_iter

    def eval(self, *args):

        # Get Input arguments in given sequence
        trn_idxs = args[0]
        crv_idxs = args[1]
        corpus_mtrx = args[2]
        cls_gnr_tgs = args[3]
        vocab_index_dct = args[4]  # tid
        params = args[5]

        if self.bagging:
            # Execute predict() with 'trn_idxs' and 'cls_gnr_tgs' arguments which triggers...
            # ...Bagging mode of RFSE
            results = self.predict(
                crv_idxs, corpus_mtrx, cls_gnr_tgs, vocab_index_dct, params, trn_idxs
            )

        else:
            # Build Genre Classes given the training vectors
            gnr_classes = self.contruct_classes(trn_idxs, corpus_mtrx, cls_gnr_tgs)

            # Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
            results = self.predict(
                crv_idxs, corpus_mtrx, cls_gnr_tgs, vocab_index_dct, params, gnr_classes
            )

        # Expected Results for the ParamGridCrossValBase class in paramgridcrossval module
        predicted_Y = results[0]
        predicted_scores = results[1]
        model_specific_d = {
            'max_sim_scores_per_iter': results[2],
            'predicted_classes_per_iter': results[3]
        }

        # Return results as expected form ParamGridCrossValBase class
        return predicted_Y, predicted_scores, model_specific_d


def cosine_similarity(vector, centroid):

    # Convert Arrays and Matrices to Matrix Type for preventing unexpected error, such as...
    # ...returning vector instead of scalar
    vector = np.matrix(vector)
    centroid = np.matrix(centroid)

    return vector * np.transpose(centroid) / (np.linalg.norm(vector) * np.linalg.norm(centroid))


def cosine_similarity_sparse(vector, centroid):

    return cosine_similarity(vector.todense(), centroid)


def minmax_similarity(v1, v2):

    return np.sum(np.min(np.vstack((v1, v2)), axis=0)) / np.sum(np.max(np.vstack((v1, v2)), axis=0))


def jaccard_similarity_binconv(v0, v1):

    v0 = np.where((v0 > 0), 1, 0)
    v1 = np.where((v0 > 0), 1, 0)

    return 1.0 - spd.jaccard(v0, v1)


def hamming_similarity(vector, centroid):

    return 1.0 - spd.hamming(centroid, vector)


def correlation_similarity(vector, centroid):

    vector = vector[0]
    centroid = np.array(centroid)[0]

    vector_ = np.where(vector > 0, 0, 1)
    centroid_ = np.where(centroid > 0, 0, 1)

    s11 = np.dot(vector, centroid)
    s00 = np.dot(vector_, centroid_)
    s01 = np.dot(vector_, centroid)
    s10 = np.dot(vector, centroid_)

    denom = np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))
    if denom == 0.0:
        denom = 1.0

    return (s11*s00 - s01*s10) / denom
