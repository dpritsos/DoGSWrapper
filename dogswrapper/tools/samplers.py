
#
# Tools for spliting the Samples into Folds and Openness Splits
#

import cPickle as pickle
import numpy as np


def SelectStratifiedKfolds(smpls_num, kfolds):

    tS_splt_lst = list()
    Tr_splt_lst = list()

    if type(smpls_num) == int:

        smpl_idxs_vect = np.arange(smpls_num)
        smpl_idxs_choice = np.arange(smpls_num)
        tst_splt_size = int(np.ceil(smpls_num/float(kfolds)))

    else type(smpls_num) is list:

        smpl_idxs_vect = smpls_num
        smpl_idxs_choice = smpls_num
        tst_splt_size = int(np.ceil(len(smpls_num)/float(kfolds)))

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


def OpenSetUNoiseSpltSamples(cls_tgs_lst, marked_uknw_ctg_lst, kfolds):

    if not len(uknw_ctg_lst):
        raise Exception("At lest one class-tag shoud be given as argument")

    knwn_ctgs_idxs_lst = list()
    uknw_ctgs_idxs_lst = list()

    for i, tag in enumerate(cls_gnr_tgs):

        if tag in uknw_ctg_lst:
            marked_uknw_ctgs_idxs_lst.append(i)
        else:
            knwn_ctgs_idxs_lst.append(i)

    # Creating the Stratifies K-Folds using only the known indeces.
    trn_slst, tst_slst = SelectStratifiedKfolds(knwn_ctgs_idxs_lst, kfolds)

    # Appending the uknown indeces
    tst_slst = [tst_l.extend(marked_uknw_ctgs_idxs_lst) for tst_l in tst_slst]

    # Returning the Training and Test Splits together with cls_tags where Known-Noise indeces...
    # ...have been marked with '0' class tag, i.e. unknown-noise tag.
    return trn_slst, tst_slst, marked_uknw_ctgs_idxs_lst


def OpennessSplitSamples(cls_tgs_lst, onlytest_clsnum, uknw_ctgs_num_splt_itrs, kfolds):

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
            tr_iidx_lst, ts_iidx_lst = SelectStratifiedKfolds(
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


def SaveSplitSamples(train_splts, test_splts, onlyt_splts, ukn_cls_num, ukn_iters, save_path):

    # Saving the splits.
    splt_fname_suffix = '_S' + str(ukn_cls_num) + '_I' + str(ukn_iters)

    trn_fname = save_path + 'Training_Splits' + splt_fname_suffix + '.pkl'

    with open(trn_fname, 'w') as f:
        pickle.dump(train_splts, f)

    test_fname = save_path + 'Testing_Splits' + splt_fname_suffix + '.pkl'
    with open(test_fname, 'w') as f:
        pickle.dump(test_splts, f)

    onlytest_fname = save_path + 'OnlyTesting_Splits' +\
        splt_fname_suffix + '.pkl'
    with open(onlytest_fname, 'w') as f:
        pickle.dump(onlyt_splts, f)


def LoadSplitSamples(ukn_cls_num, ukn_iters, save_path):

    # Saving the splits.
    splt_fname_suffix = '_S' + str(ukn_cls_num) + '_I' + str(ukn_iters)

    trn_fname = save_path + 'Training_Splits' + splt_fname_suffix + '.pkl'

    with open(trn_fname, 'r') as f:
        train_splts = pickle.load(f)

    test_fname = save_path + 'Testing_Splits' + splt_fname_suffix + '.pkl'
    with open(test_fname, 'r') as f:
        test_splts = pickle.load(f)

    onlytest_fname = save_path + 'OnlyTesting_Splits' +\
        splt_fname_suffix + '.pkl'
    with open(onlytest_fname, 'r') as f:
        onlyt_splts = pickle.load(f)

    return train_splts, test_splts, onlyt_splts
