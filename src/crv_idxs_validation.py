

import json
import numpy as np


crv_filename = '/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_4Grams/kfold_crv_0.idx'
#crv_filename = '/home/dimitrios/Synergy-Crawler/SANTINIS/kfold_crv_0.idx'
tags_lst = '/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_4Grams/Corpus_tags_shorted.lst'

with open(tags_lst, 'r') as f:
	cls_gnr_tgs = json.load(f, encoding='utf-8')

with open(crv_filename, 'r') as f:
	crv = json.load(f)

#crv = np.array(crv)
print crv
cls_gnr_tgs = np.array(cls_gnr_tgs)
print cls_gnr_tgs[crv]

print len(crv)
print len(np.unique(crv))
print crv

u, i = np.unique(crv, return_inverse=True)
print u[ np.bincount(i) > 1], len( u[ np.bincount(i) > 1] )



