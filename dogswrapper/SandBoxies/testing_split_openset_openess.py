
import numpy as np
from sklearn import cross_validation


def select_stratified_kfolds(smpls_num, kfolds):

    tS_splt_lst = list()
    Tr_splt_lst = list()
    tst_splt_size = int(np.ceil(smpls_num/float(kfolds)))

    smpl_idxs_vect = np.arange(smpls_num)
    smpl_idxs_choice = np.arange(smpls_num)

    for k in np.arange(kfolds):

        # If the samples choice list is not empty. This option is for preveting the for...
        # ...loop to return error when the samples number is too small.
        if smpl_idxs_choice.shape[0] == 0:
            break # smpl_idxs_choice = np.arange(smpls_num)

        test_smpls = np.random.choice(smpl_idxs_choice, tst_splt_size, replace=False)

        smpl_idxs_choice = smpl_idxs_choice[np.in1d(smpl_idxs_choice, test_smpls, invert=True)]

        Tr_splt_lst.append(smpl_idxs_vect[np.in1d(smpl_idxs_vect, test_smpls, invert=True)])
        tS_splt_lst.append(test_smpls)

    return Tr_splt_lst, tS_splt_lst


def OpennessSplitSamples(cls_tgs_lst, onlytest_clsnum, onlytest_splt_itrs, kfolds):

    # Converting to numpy.array for compatibility with array operations.
    cls_tgs_lst = np.array(cls_tgs_lst)

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
        onlytest_csampls_mask = np.in1d(cls_tgs_lst, onlytest_clstags)

        # Getting the classes-samples indeces bind only for testing splits.
        onlytest_csampls_idxs = np.where(onlytest_csampls_mask == True)[0]

        # Getting the classes-samples to be used for Kfold training/testing spliting.
        tt_csampls_idxs = np.where(onlytest_csampls_mask == False)[0]

        # Getting the class-tags (per sapmles) which will be used for training/testing spliting.
        tt_cls_tgs_lst = cls_tgs_lst[tt_csampls_idxs]

        Tr_kfmatrx_per_cls, tS_kfmatrx_per_cls = list(), list()

        for ctg in np.unique(tt_cls_tgs_lst):

            # Getting the class-samples indeces.
            this_cls_idxs = np.where(tt_cls_tgs_lst == ctg)[0]
            # print this_cls_idxs

            # Statified Kfold Selection of samples in training and test sets.
            tr_iidx_lst, ts_iidx_lst = select_stratified_kfolds(this_cls_idxs.shape[0], kfolds)

            Tr_kfmatrx_per_cls.append(
                np.vstack(
                    [tt_csampls_idxs[this_cls_idxs[tr_iidx]] for tr_iidx in tr_iidx_lst]
                )
            )

            tS_kfmatrx_per_cls.append(
                np.vstack(
                    [tt_csampls_idxs[this_cls_idxs[ts_iidx]] for ts_iidx in ts_iidx_lst]
                )
            )

        Tr_kfs_4_osplts.append(np.hstack(Tr_kfmatrx_per_cls))
        tS_kfs_4_osplts.append(
            np.hstack((
                np.hstack(tS_kfmatrx_per_cls),
                np.vstack([onlytest_csampls_idxs for i in range(tS_kfmatrx_per_cls[0].shape[0])])
            ))
        )

    return Tr_kfs_4_osplts, tS_kfs_4_osplts


Tr_kfs_4_osplts, tS_kfs_4_osplts = OpennessSplitSamples([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5], 1, 3, 15)

print Tr_kfs_4_osplts
print
print tS_kfs_4_osplts



#l = list()
#for i in select_stratified_kfolds(100, 10):
# l.append(i[1])

# print np.sort(np.hstack(l))
