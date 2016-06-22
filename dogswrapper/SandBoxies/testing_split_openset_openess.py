
import numpy as np
from sklearn import cross_validation


def select_stratified_kfolds(smpls_num, kfolds):

    tS_splt_lst = list()
    Tr_splt_lst = list()
    tst_splt_size = int(np.ceil(smpls_num/float(kfolds)))

    smpl_idxs_vect = np.arange(smpls_num)
    smpl_idxs_choice = np.arange(smpls_num)

    for k in np.arange(kfolds):

        test_smpls = np.random.choice(smpl_idxs_choice, tst_splt_size, replace=False)

        smpl_idxs_choice = smpl_idxs_choice[np.in1d(smpl_idxs_choice, test_smpls, invert=True)]

        Tr_splt_lst.append(smpl_idxs_vect[np.in1d(smpl_idxs_vect, test_smpls, invert=True)])
        tS_splt_lst.append(test_smpls)

    return Tr_splt_lst, tS_splt_lst


def OpennessSplitSamples(cls_tgs_lst, onlytest_clsnum, onlytest_splt_itrs, kfolds):

    # Two list of arrays where each array has the file indeces for training and testing....
    # ...Each list entry containts the kFold cross-validation splits for a random selection...
    # ...of class-tags remaining Only-For-Testing. The Only-For-Testing class samples are...
    # ...appended at the end of each test-samples-split in every kfold.
    Tr_kfs_4_osplts, tS_kfs_4_osplts = list(), list()

    # Starting Openness Random Selection Class Spliting.
    for i in range(onlytest_splt_itrs):

        # Selecting the Class tags tto be excluded from traing set kfold splits.
        onlytest_clstags = np.random.choice(
            np.unique(cls_tgs_lst), onlytest_clsnum, replace=False
        )

        # Getting the mask for class-samples to be excluded from traing set kfold splits.
        onlytest_csampls_mask = np.in1d(onlytest_clstags, cls_tgs_lst)

        # Getting the classes-samples indeces bind only for testing splits.
        onlytest_csampls_idxs = np.where(onlytest_csampls_mask == True)

        # Getting the classes-samples to be used for Kfold training/testing spliting.
        tt_csampls_idxs = np.where(onlytest_csampls_mask == False)

        # Getting the class-tags (per sapmles) which will be used for training/testing spliting.
        tt_cls_tgs_lst = cls_tgs_lst[tt_csampls_idxs]

        Tr_kfmatrx_per_cls, tS_kfmatrx_per_cls = list(), list()

        for ctg in np.unique(cls_tgs_lst):

            # Getting the class-samples indeces.
            this_cls_idxs = np.where(cls_tgs == ctg)[0]

            tr_iidx_lst, ts_iidx_lst = select_stratified_kfolds(this_cls_idxs.shape[0], kfolds)


            Tr_kfmatrx_per_cls.append(
                np.vstack(
                    [this_cls_idxs[tr_iidx] for tr_iidx in tr_iidx_lst]
                )
            )
            tS_kfmatrx_per_cls.append(
                np.vstack(
                    [this_cls_idxs[ts_iidx] for ts_iidx in ts_iidx_lst]
                )
            )



    for ctg in np.unique(cls_tgs):



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


    return train_subsplits_arrlst, testing_subsplits_arrlst




l = list()
for i in select_stratified_kfolds(100, 10):
    l.append(i[1])

print np.sort(np.hstack(l))
