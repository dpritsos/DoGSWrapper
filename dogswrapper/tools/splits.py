
#
# Tools for spliting the Samples into Folds and Openness Splits
#

import numpy as np


def SelectStratifiedKfolds(self, smpls_num, kfolds):

    tS_splt_lst = list()
    Tr_splt_lst = list()
    tst_splt_size = int(np.ceil(smpls_num/float(kfolds)))

    smpl_idxs_vect = np.arange(smpls_num)
    smpl_idxs_choice = np.arange(smpls_num)

    for k in np.arange(kfolds):

        # If the samples choice list is not empty. This option is for preveting the...
        # ...loop to return error when the samples number is too small.
        # if smpl_idxs_choice.shape[0] == 0:
        #    print "Small"
        #    break  # smpl_idxs_choice = np.arange(smpls_num)

        # Stopping the look when samples amount is smaller than samples required...
        # ...to be selected. Thus, repeating a few indeces at random just for satisfing the...
        # random.choice() function.
        if len(smpl_idxs_choice) < tst_splt_size:
            apnd_sel = smpl_idxs_vect[np.in1d(smpl_idxs_vect, smpl_idxs_choice, invert=True)]
            apnd_size = tst_splt_size - len(smpl_idxs_choice)
            apnd_idxs = np.random.choice(apnd_sel, apnd_size, replace=False)
            test_smpls = np.hstack((smpl_idxs_choice, apnd_idxs))
        else:
            test_smpls = np.random.choice(smpl_idxs_choice, tst_splt_size, replace=False)

        smpl_idxs_choice = smpl_idxs_choice[np.in1d(smpl_idxs_choice, test_smpls, invert=True)]

        Tr_splt_lst.append(smpl_idxs_vect[np.in1d(smpl_idxs_vect, test_smpls, invert=True)])
        tS_splt_lst.append(test_smpls)

    return Tr_splt_lst, tS_splt_lst


def OpennessSplitSamples(self, cls_tgs_lst, onlytest_clsnum, uknw_ctgs_num_splt_itrs, kfolds):

    # Converting to numpy.array for compatibility with array operations.
    cls_tgs_lst = np.array(cls_tgs_lst)

    # Two list of arrays where each array has the file indeces for training and testing....
    # ...Each list entry containts the kFold cross-validation splits for a random selection...
    # ...of class-tags remaining Only-For-Testing. The Only-For-Testing class samples are...
    # ...appended at the end of each test-samples-split in every kfold.
    Tr_kfs_4_osplts, tS_kfs_4_osplts, oT_kfs_4_osplts = list(), list(), list()

    # Starting Openness Random Selection Class Spliting.
    for i in range(uknw_ctgs_num_splt_itrs):

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
            tr_iidx_lst, ts_iidx_lst = self.SelectStratifiedKfolds(
                this_cls_idxs.shape[0], kfolds
            )

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
                np.vstack(
                    [onlytest_csampls_idxs for i in range(tS_kfmatrx_per_cls[0].shape[0])]
                )
            ))
        )
        oT_kfs_4_osplts.append(
            np.vstack(
                [onlytest_csampls_idxs for i in range(tS_kfmatrx_per_cls[0].shape[0])]
            )
        )
    return Tr_kfs_4_osplts, tS_kfs_4_osplts, oT_kfs_4_osplts


def SaveSplitSamples(self, train_subsplits_arrlst, testing_subsplits_arrlst, ot_subsp_arrlst,
                     fnames_tpl=('Trning_S.pkl', 'Tsting_Splts.pkl', 'OTing_Splts.pkl'),
                     process_state_saving_path=None):

    # Replace the default or create if required the path where Sample Splits will be saved.
    if process_state_saving_path:
        save_path = process_state_saving_path
    else:
        save_path = self.state_save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Set the file names for training and testing splits.
    train_splits_path = save_path + fnames_tpl[0]
    test_splits_path = save_path + fnames_tpl[1]
    only_test_splits_path = save_path + fnames_tpl[2]

    # Pickleing the Training and Testing Splits.
    with open(train_splits_path, 'w') as f:
        pickle.dump(train_subsplits_arrlst, f)

    with open(test_splits_path, 'w') as f:
        pickle.dump(testing_subsplits_arrlst, f)

    with open(only_test_splits_path, 'w') as f:
        pickle.dump(ot_subsp_arrlst, f)


def LoadSplitSamples(self, fnames_tpl=('Trning_S.pkl', 'Tsting_Splts.pkl', 'OTing_Splts.pkl'),
                     process_state_saving_path=None):

    # Replace the path where the process-state files was supposed to be saved.
    if process_state_saving_path:
        save_path = process_state_saving_path
    else:
        save_path = self.state_save_path

    if not os.path.exists(save_path):
        raise Exception(
            "Loading Samples Splits Faild: process-state-saving-path does not exist"
        )

    # Set the file names for training and testing splits.
    train_splits_fname = save_path + fnames_tpl[0]
    test_splits_fname = save_path + fnames_tpl[1]
    only_test_splits_fname = save_path + fnames_tpl[2]

    if os.path.exists(train_splits_fname) and os.path.exists(test_splits_fname):

        # Unpickleing the Training and Testing Splits.
        with open(train_splits_fname, 'r') as f:
            train_subsplits_arrlst = pickle.load(f)

        with open(test_splits_fname, 'r') as f:
            testing_subsplits_arrlst = pickle.load(f)

        with open(only_test_splits_fname, 'r') as f:
            only_testing_subsplits_arrlst = pickle.load(f)

    else:

        return None, None, None

    return train_subsplits_arrlst, testing_subsplits_arrlst, only_testing_subsplits_arrlst
