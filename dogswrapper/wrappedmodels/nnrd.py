
import numpy as np
import scipy.spatial.distance as spd
import sys

sys.path.append('../../')
import OpenNNDR.onndr.onndr as onndr


def nnrd_eval(*args):

    # Get Input arguments in given sequence
    trn_idxs = args[0]
    crv_idxs = args[1]
    corpus_mtrx = args[2]
    cls_gnr_tgs = args[3]
    params = args[4]

    # Initilising the NN model.
    osnndr = onndr.OpenNNDR(
        slt_ptg=params['split_ptg'],
        ukwn_slt_ptg=params['ukwn_slt_ptg'],
        rt_lims_stp=params['rt_lims_stp'],
        lmda=params['lmda']
    )

    # Optimising the rt threshold based on the Training Data.
    cls_d, rt = osnndr.fit(corpus_mtrx[trn_idxs, :], cls_gnr_tgs[trn_idxs])

    # Executing predict() and getting the predicted Class Tags. 0 means Unclassified!
    pre_y, pre_r = osnndr.predict(corpus_mtrx[crv_idxs, :])

    # Returning the results.
    return pre_y, pre_r, rt
