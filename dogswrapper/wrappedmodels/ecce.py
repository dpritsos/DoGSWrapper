
import numpy as np
import sys

sys.path.append('../../')
import ECCE as ecce


def eval(*args):

    # Get Input arguments in given sequence
    trn_idxs = args[0]
    crv_idxs = args[1]
    corpus_mtrx = args[2]
    cls_gnr_tgs = args[3]
    params = args[4]

    # Initilising the RFSE model.
    ecce_mdl = ecce.ECCE(
        sim_func=params['sim_func'],
        inpt_vectz=1,
        agg_comb='product'
    )

    # Fitting the data to the models. Accuatlly caclulating the Class Centroids in this case.
    gnr_clz = ecce_mdl.fit(corpus_mtrx[trn_idxs, :], cls_gnr_tgs[trn_idxs])

    # Executing predict() and getting the predicted Class Tags. 0 means Unclassified!
    # print crv_idxs
    # print corpus_mtrx
    # print corpus_mtrx[crv_idxs, :]
    res = ecce_mdl.predict(corpus_mtrx[crv_idxs, :])

    # Returning the results.
    return {
        'predicted_Y': res[0],
        'predicted_scores': res[1],
        # 'gnr_csums': res[2],
        # 'gnr_ssums': res[3],
        # 'grn_snum': res[4]
    }
