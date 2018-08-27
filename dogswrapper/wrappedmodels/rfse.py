
import numpy as np
import RFSE as rfse


def eval(*args):

    # Get Input arguments in given sequence
    trn_idxs = args[0]
    crv_idxs = args[1]
    corpus_mtrx = args[2]
    cls_gnr_tgs = args[3]
    params = args[4]

    # Initilising the RFSE model.
    rfse_mdl = rfse.RFSE(
        sim_func=params['sim_func'],
        itrs=params['Iterations'],
        sigma=params['Sigma'],
        feat_size=params['features_size'],
        bagging=0.0
    )

    # Fitting the data to the models. Accuatlly caclulating the Class Centroids in this case.
    gnr_clz = rfse_mdl.fit(corpus_mtrx[trn_idxs, :], cls_gnr_tgs[trn_idxs])

    # Executing predict() and getting the predicted Class Tags. 0 means Unclassified!
    print crv_idxs
    print corpus_mtrx
    print corpus_mtrx[crv_idxs, :]
    res = rfse_mdl.predict(corpus_mtrx[crv_idxs, :])

    # Returning the results.
    return {
        'predicted_Y': res[0],
        'predicted_scores': res[1],
        'max_sim_scores_per_iter': res[2],
        'predicted_classes_per_iter': res[3]
    }
