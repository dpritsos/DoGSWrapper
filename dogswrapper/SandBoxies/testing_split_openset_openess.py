
import numpy as np
from sklearn import cross_validation


def SplitSamples(cls_tgs, trn_percent=0.5, decrease_step=0.1, method='rndred_trn_fixed_test'):

    # Checking trn_percent and decrease_step value constraints.
    if trn_percent < 0.001 or trn_percent > 1.0 or decrease_step < 0.001 or decrease_step > 1.0:
        raise Exception("trm_percent and decrease_step values mast be in range [0.001, 1]")

    # Two list of arrays where each array has the file indeces for training and testing...
    # ...repspectivly splitted initially upon trn_percentage.
    trn_splts_per_ctg_arrlst, tst_splts_per_ctg_arrlst = list(), list()

    for ctg in np.unique(cls_tgs):

        # Getting the filename list indeces for this class (tag).
        this_cls_idxs = np.where(cls_tgs == ctg)[0]

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

    # Two lists per sub-split one for training and one for testing. Every element of the...
    # ...list is containing an array where the rows are containing the training and...
    # ...testing index splits for every class (tag) respectively.
    train_subsplits_arrlst, testing_subsplits_arrlst = list(), list()

    for trn_decreased_perc in np.arange(trn_percent, 0.0, -decrease_step):

        train_ctg_lst, test_ctg_lst = list(), list()

        for trn_arr, tst_arr in zip(trn_splts_per_ctg_arrlst, tst_splts_per_ctg_arrlst):

            smpls_num = int(
                np.ceil(this_cls_idxs.shape[0] * trn_decreased_perc)
            )

            # Selecting the method to split the corpus to training and test sets.
            if method == 'rndred_trn_fixed_test':

                # Keeping only a partition of the training indeces split, while the...
                # ...testning split remains the same.
                train_ctg_lst.append(trn_arr[0:smpls_num])
                test_ctg_lst.append(tst_arr)

            elif method == 'rndred_trn_rest4_test':

                # Keeping only a partition of the training indeces split, while the...
                # ...testing split is extended with the rest of the training split.
                train_ctg_lst.append(trn_arr[0:smpls_num])
                test_ctg_lst.append(
                    np.short(
                        np.hstack(
                            (tst_arr, trn_arr[smpls_num::])
                        )
                    )
                )

            else:
                raise Exception("Non-implemented yet!")

        # Keeping the sub-splits array lists.
        train_subsplits_arrlst.append(
            np.vstack(train_ctg_lst)
        )

        testing_subsplits_arrlst.append(
            np.vstack(test_ctg_lst)
        )

    return train_subsplits_arrlst, testing_subsplits_arrlst


def select_stratified_kfolds(smpls_num, kfolds):

    k_tt_splts_lst = list()
    tst_splt_size = int(np.ceil(smpls_num/float(kfolds)))

    smpl_idxs_vect = np.arange(smpls_num)
    smpl_idxs_choice = np.arange(smpls_num)

    for k in np.arange(kfolds):

        test_smpls = np.random.choice(smpl_idxs_choice, tst_splt_size)

        smpl_idxs_choice = smpl_idxs_choice[np.where(smpl_idxs_choice != test_smpls)]

        k_tt_splts_lst.append(
            (smpl_idxs_vect[np.where(smpl_idxs_vect != test_smpls)], test_smpls)
        )

    return k_tt_splts_lst


for i in select_stratified_kfolds(10, 10):
    print i[0], i[1]
