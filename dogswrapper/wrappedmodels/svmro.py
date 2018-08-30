
import SVMRO as svmro


def eval(*args):

    # Get Input arguments in given sequence
    trn_idxs = args[0]
    crv_idxs = args[1]
    corpus_mtrx = args[2]
    cls_gnr_tgs = args[3]
    params = args[4]

    if params['svm_type'] == 'oneclass':
        svmro_i = svmro.SVMRO(
            svm_type='oneclass',
            ll=params['ll'],
            c1_w=params['c1_w'], c2_w=params['c2_w'],
            mrgn_nw=params['mrgn_nw'], mrgn_fw=params['mrgn_fw'],
            nu=params['nu']
        )
    else:
        svmro_i = svmro.SVMRO(
            svm_type='binary',
            ll=params['ll'],
            c1_w=params['c1_w'], c2_w=params['c2_w'],
            mrgn_nw=params['mrgn_nw'], mrgn_fw=params['mrgn_fw'],
        )

    # Build Genre Classes given the training vectors
    print trn_idxs.shape, corpus_mtrx.shape
    svmro_i.fit(corpus_mtrx[trn_idxs, :], cls_gnr_tgs[trn_idxs])

    # Execute predict() with gnr_classes which triggers simple RFSE (non Bagging)
    pre_y, pDn_per_gnr, pDf_per_gnr, pre_maxD_scores = svmro_i.predict(corpus_mtrx[crv_idxs, :])

    # Returning the results.
    return {
        'predicted_Y': pre_y,
        'predicted_Dn_per_gnr': pDn_per_gnr,
        'predicted_Df_per_gnr': pDf_per_gnr,
        'predicted_maxD_score': pre_maxD_scores
    }
