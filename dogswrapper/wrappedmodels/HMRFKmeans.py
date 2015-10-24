

import numpy as np
import sys
sys.path.append('../../Djumble/')
from djumble.vmf_semisupervised_kmeans import HMRFKmeans

class HMRFKmeans_Wrapped(object):

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

    def DoSemiSupervdClustrering(self, trn_subsplt, tst_subsplt, corpus_mtrx, params):

        print "DIMS: ", corpus_mtrx.shape[1]
        # print params
        # 0/0
        # Building the Must-Link and Cannot-Link Constraints.
        must_lnk, cannot_lnk = self.BuildContraints(trn_subsplt)

        # Getting the number of the expected clusters.
        k_clusters = trn_subsplt.shape[0]
        print "Clusters (Expected): ", k_clusters

        # Selecting randomly a set of initial centroids, i.e., one index from every class.
        init_centrs = [
            set(np.random.choice(cls_row_idxs, 1)) for cls_row_idxs in trn_subsplt
        ]

        # Initializing the HMRF Kmeans Semi-Supervised Clustering Model upon params argument and...
        # k-clusters expected, Must-Link and Cannot-Link constraints.
        print corpus_mtrx.shape[1]
        hkmeans = HMRFKmeans(
            k_clusters,  must_lnk, cannot_lnk, init_centroids=init_centrs,
            max_iter=params['max_iter'], cvg=params['converg_diff'],
            lrn_rate=params['learing_rate'], ray_sigma=1.0,
            w_violations=np.random.uniform(
                0.3, 0.3, size=(corpus_mtrx.shape[0], corpus_mtrx.shape[0])),
            d_params=np.random.uniform(1.0, 1.0, size=corpus_mtrx.shape[1]),
            norm_part=True, globj='non-normed'
        )

        # Serializing the training split indeces.
        srl_trn_spl = trn_subsplt.reshape((1, np.multiply(*trn_subsplt.shape)))
        srl_tst_spl = tst_subsplt.reshape((1, np.multiply(*tst_subsplt.shape)))

        # Getting the Indeced split for clustering by Stacking them and sorting the above.
        subset_split_idxs = np.short(np.hstack((srl_trn_spl, srl_tst_spl)))[0]

        # Doing the Semi-Supervised Clustering for this Corpus Split.
        res = hkmeans.fit(corpus_mtrx[subset_split_idxs, :])

        # Converting the list index sets to vector of cluster tags.
        # NOTE: mu_lst, clstr_idxs_set_lst, self.A.data  = res

        # Setting the place holder for the clusters-tag vector to be returned.
        clstr_tags_arr = np.zeros(corpus_mtrx.shape[0], dtype=np.int32)

        # Assigning the cluster tags per vector position.
        for i, iset in res[1]:
            for idx in iset:
                clstr_tags_arr[idx] = i+1

        # Keeping only the non-zero values which is the respective sample indeces given for...
        # ...clustering from the whole corpus.
        clstr_tags_arr_nonzero = clstr_tags_arr[np.where(clstr_tags_arr != 0)]

        # model_specific_d = {
        #     'max_sim_scores_per_iter': results[2],
        #     'predicted_classes_per_iter': results[3]
        # }

        # Return results.
        return clstr_tags_arr_nonzero
