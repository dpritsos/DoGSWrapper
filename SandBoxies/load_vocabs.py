

import pickle



with open('/media/dimitrios/dbbaa90d-fc43-4e53-84ca-e5aa1bce18d9/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_Word_1Grams/kfold_Voc_1.pkl', 'r') as f:
	old_tf_d = pickle.load(f)
	print min(old_tf_d.values()), max(old_tf_d.values())
	print old_tf_d.keys()[0:10]


with open('/media/dimitrios/dbbaa90d-fc43-4e53-84ca-e5aa1bce18d9/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_Word_1Grams_New/kfold_Voc_1.pkl', 'r') as f:
	new_tf_d = pickle.load(f)
	print min(new_tf_d.values()), max(new_tf_d.values())
	print new_tf_d.keys()[0:10]

	ft_d = dict( zip(new_tf_d.values(), new_tf_d.keys()) )
	print ft_d[ max(new_tf_d.values()) ]





