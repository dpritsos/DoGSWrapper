
# #### TESTIng ###############


import numpy as np


class HMRFKmeans_Wrapped(object):

    def __init__(self):
        pass

    def SplitSample(self, cls_tgs, trn_percent=0.5, decrease_step=0.1, method='on-file-prefix'):

        # Checking trn_percent and decrease_step value constraints.
        if trn_percent < 0.01 or trn_percent > 1.0 or decrease_step < 0.01 or decrease_step > 1.0:
            raise Exception("trm_percent and decrease_step values mast be in range [0.01, 1]")

        # Selecting the method to split the corpus to training and test sets.
        if method == 'on-file-prefix':

            # Two list of arrays where each array has the file indeces for training and testing...
            # ...repspectivly splitted initially upon trn_percentage.
            trn_splts_per_ctg_arrlst, trn_splts_per_ctg_arrlst = list(), list()

            for ctg in np.unique(cls_tgs):

                # Getting the filename list indeces for this class (tag).
                this_cls_idxs = np.where(cls_tgs == ctg)[0]

                # Calculating the amount of samples keeping for training for this class for the...
                # ...initial split.
                smpls_num = np.ceil(this_cls_idxs.shape[0] * trn_percent)

                # NOTE: Here the training indeces are selected Randomly! Thus, it is easy to...
                # ...use this python-class-method into Cross-Validation experimental set-up....
                # ...The list of indeces is sorted.
                train_files_idxs_arr = np.sort(
                    np.random.choice(this_cls_idxs, smpls_num, replace=False)
                )

                trn_splts_per_ctg_arrlst.append(train_files_idxs_arr)

                # Keeping the indeces but the ones for training as testing indeces.
                trn_splts_per_ctg_arrlst.append(
                    np.array(
                        [tst_i for tst_i in this_cls_idxs if tst_i not in train_files_idxs_arr]
                    )
                )

            # Two lists per sub-split one for training and one for testing. Every element of the...
            # ...list is containing an array where the rows are containing the training and...
            # ...testing index splits for every class (tag) respectively.
            train_subsplits_arrlst, testing_subsplits_arrlst = list(), list()

            for splts_decreased_perc in np.arange(1, decrease_step, -decrease_step):

                train_ctg_lst, test_ctg_lst = list(), list()

                for trn_splts_per_ctg_arr in trn_splts_per_ctg_arrlst:

                    smpls_num = np.ceil(trn_splts_per_ctg_arr.shape[0] * splts_decreased_perc)

                    train_ctg_lst.append(trn_splts_per_ctg_arr[0:smpls_num])
                    test_ctg_lst.append(trn_splts_per_ctg_arrlst)

                # Keeping the sub-splits array lists.
                train_subsplits_arrlst.append(
                    np.vstack(train_ctg_lst)
                )

                testing_subsplits_arrlst.append(
                    np.vstack(test_ctg_lst)
                )

        else:
            raise Exception("Non-implemented yet!")

        return train_subsplits_arrlst, testing_subsplits_arrlst

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

if __name__ == '__main__':

    cls_tgs = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    )

    test = HMRFKmeans_Wrapped()

    train_splits_lst, test_splits_lst = test.SplitSampleBackets(cls_tgs, trn_percent=0.5, decrease_step=0.1, method='on-file-prefix')
    """
    for split_no, splits_per_class in enumerate(train_splits_lst[0]):
        print 'Splits No.', split_no
        print splits_per_class
        print
    """

    print len(train_splits_lst)
    print train_splits_lst[0]
    print train_splits_lst[5]

    #must_lnk, cannot_lnk = test.BuildContraints(train_splits_lst[3])

    #for i, pair in enumerate(cannot_lnk):
    #    print i, pair
