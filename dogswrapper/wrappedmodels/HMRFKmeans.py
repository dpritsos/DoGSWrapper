
import tables
import numpy as np
import sys
sys.path.append('../../')
from Djumble.djumble.semisupkmeans import HMRFKmeansSemiSup




class HMRFKmeansSemiSup_Wrapped(object):

    def __init__(self):
        pass

    def BuildContraints(self, train_ctg_idxs_arr):

        # Building must-link constraints.
        must_lnk = list()

        for train_ctg_idxs_row in train_ctg_idxs_arr:

            for i, idx1 in enumerate(train_ctg_idxs_row[0:-1]):

                for idx2 in train_ctg_idxs_row[i+1::]:

                    must_lnk.append(set([idx1, idx2]))

        # Building cannot-link constraints.
        cannot_lnk = list()
        for i, train_ctg_idxs_row in enumerate(train_ctg_idxs_arr[0:-1, :]):

            for train_ctg_idxs_row_next in train_ctg_idxs_arr[i+1::, :]:

                cannot_lnk.extend([
                    set([idx1, idx2]) for idx1 in train_ctg_idxs_row
                    for idx2 in train_ctg_idxs_row_next
                ])

        return must_lnk, cannot_lnk

    def MakeSetsArrays(self, cons_lst_set):

        cons_arr = np.zeros((2, len(cons_lst_set)), dtype=np.int)

        for i, iset in enumerate(cons_lst_set):
            cons_arr[0, i] = iset.pop()
            cons_arr[1, i] = iset.pop()

        return cons_arr

    def DoSemiSupervdClustrering(self, trn_subsplt, tst_subsplt, corpus_mtrx, params):

        if isinstance(corpus_mtrx, tables.EArray):
            print "DIMS: ", corpus_mtrx.read().shape[1]
        else:
            print "DIMS: ", corpus_mtrx.shape[1]
        # print params
        # 0/0
        # Building the Must-Link and Cannot-Link Constraints.
        must_lnk, cannot_lnk = self.BuildContraints(trn_subsplt)

        # Only for Narray versions!!!
        must_lnk = self.MakeSetsArrays(must_lnk)
        cannot_lnk = self.MakeSetsArrays(cannot_lnk)

        print must_lnk
        print cannot_lnk

        # print 'ml & cl cons count', len(must_lnk), len(cannot_lnk)
        print 'ml & cl cons count', must_lnk.shape[1], cannot_lnk.shape[1]

        # Getting the number of the expected clusters.
        k_clusters = trn_subsplt.shape[0]
        print "Clusters (Expected): ", k_clusters

        # Selecting randomly a set of initial centroids, i.e., one index from every class.
        # init_centrs = [
        #    set(np.random.choice(cls_row_idxs, 1)) for cls_row_idxs in trn_subsplt
        # ]
        # Only in case of Narray!!!
        init_centrs = [
            np.random.choice(cls_row_idxs, 1) for cls_row_idxs in trn_subsplt
        ]
        # init_centrs = np.array(init_centrs)
        # For Cyhton version.
        init_centrs = np.hstack(np.array(init_centrs))

        # Initializing the HMRF Kmeans Semi-Supervised Clustering Model upon params argument and...
        # k-clusters expected, Must-Link and Cannot-Link constraints.

        self.hkmss = HMRFKmeansSemiSup(
            k_clusters,
            must_lnk_con_arr, cannot_lnk_con_arr,
            init_centroids=init_centrs_arr,
            ml_wg=1.0, cl_wg=1.0,
            max_iter=100,
            cvg=0.001, lrn_rate=0.003,
            ray_sigma=2.5, d_params=None,
            icm_max_i=1000,
            enable_norm=False
        )

        self.hkmeans = HMRFKmeans(
            k_clusters, must_lnk, cannot_lnk, init_centroids=init_centrs,
            ml_wg=0.30, cl_wg=0.20, max_iter=params['max_iter'],
            cvg=params['converg_diff'], lrn_rate=params['learing_rate'], ray_sigma=5.0,
            d_params=np.random.uniform(1.0, 1.0, size=corpus_mtrx.shape[1]),
            norm_part=False, globj_norm=False
        )

        if params['train_split_step_method'][2] == 'rndred_trn_fixed_test':

            # Serializing the training split indeces.
            srl_trn_spl = set(trn_subsplt.reshape((1, np.multiply(*trn_subsplt.shape)))[0])
            srl_tst_spl = set(tst_subsplt.reshape((1, np.multiply(*tst_subsplt.shape)))[0])

            # Getting the Indeced that should not participate in clustering.
            all_corp_idxs = set(range(corpus_mtrx.shape[0]))
            neg_subset_split_idxs = all_corp_idxs - srl_trn_spl - srl_tst_spl

            # Only for Narray versions!!!
            neg_subset_split_idxs = np.array(list(neg_subset_split_idxs), dtype=np.intp)

            # Doing the Semi-Supervised Clustering for this Corpus Split.
            print corpus_mtrx
            res = self.hkmeans.fit(
                np.array(corpus_mtrx, dtype=np.float64), neg_idxs4clstring=neg_subset_split_idxs
            )

        elif params['train_split_step_method'][2] == 'rndred_trn_rest4_test':

            # Doing the Semi-Supervised Clustering for this Corpus Split.
            res = self.hkmeans.fit(corpus_mtrx)

        else:
            raise Exception('Given params[train_split_step_method] option has not been defined')

        # Converting the list index sets to vector of cluster tags.
        # NOTE: mu_lst, clstr_idxs_set_lst, self.A.data  = res

        # Setting the place holder for the clusters-tag vector to be returned.
        # clstr_tags_arr = np.zeros(corpus_mtrx.shape[0], dtype=np.int)

        # Assigning the cluster tags per vector position.
        # for i, iset in enumerate(res[1]):
        #    for idx in iset:
        #        clstr_tags_arr[idx] = i+1

        # Keeping only the non-zero values which is the respective sample indeces given for...
        # ...clustering from the whole corpus.
        # clstr_tags_arr_nonzero = clstr_tags_arr[np.where(clstr_tags_arr != 0)]

        # Only for Narray versions!!!
        clstr_tags_arr = res[1]

        # Keeping only the samples indeces used clustering from the whole corpus. That is every...
        # ...sample taged with any integer other than -9 which means non-used sampel in clustering.
        clstr_tags_arr_used_only = clstr_tags_arr[np.where(clstr_tags_arr != -9)]

        # model_specific_d = {
        #     'max_sim_scores_per_iter': results[2],
        #     'predicted_classes_per_iter': results[3]
        # }

        # Return results.
        return clstr_tags_arr_used_only

    def get_params(self):
        return self.hkmeans.get_params()
