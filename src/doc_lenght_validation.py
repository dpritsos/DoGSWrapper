 

import sys
sys.path.append('../../html2vectors/src')
sys.path.append('../src')

import json
import html2vect.tables.cngrams as h2v_cng
import numpy as np
import pickle
import tables as tb

char_n_gram_size = 4
tables_cng = h2v_cng.Html2TF(char_n_gram_size, attrib='text', lowercase=True, valid_html=False)

crps_voc_path = "/home/dimitrios/Synergy-Crawler/SANTINIS/Kfolds_Vocs_Inds_4Grams"

corpus_files_lst_path = crps_voc_path+'/Corpus_filename_shorted.lst'
corpus_tags_lst_path = crps_voc_path+'/Corpus_tags_shorted.lst'
        
print "Loading HTML Filenames and Classes Tags Lists" 
    
#Load Filename and classes Tags lists
with open(corpus_files_lst_path, 'r') as f:
    html_file_l = json.load(f, encoding='utf-8')

html_file_l =  np.array(html_file_l)

trn_filename = crps_voc_path+'/kfold_trn_'+str(0)+'.idx'

with open(trn_filename, 'r') as f:
    trn = json.load(f, encoding='utf-8')
#print trn
#print np.array(trn, dtype=np.int32)

#print "Builiding Vocabulary"          
#tf_d = tables_cng.build_vocabulary( list(html_file_l[trn]), encoding='utf-8', error_handling='replace' )

pkl_voc_filename = crps_voc_path+'/kfold_Voc_'+str(0)+'.pkl'
print "Loadinging VOCABULARY for k-fold=",0
with open(pkl_voc_filename, 'r') as f:
    tf_d = pickle.load(f)

resized_tf_d = tables_cng.tfdtools.keep_atleast(tf_d, 5000) 

tid = tables_cng.tfdtools.tf2tidx( resized_tf_d )

print "Building Corpus Matrix"
corpus_mtrx_fname = crps_voc_path+'/TEST_kfold_CorpusMatrix_'+str(0)+str(5000)+'.h5'
corpus_mtrx = tables_cng.from_files(list( html_file_l ), corpus_mtrx_fname, tid_dictionary=tid, norm_func=None, encoding='utf8', error_handling='replace' )
h5f = corpus_mtrx[2]
#Load or Crreate the Coprus Matrix (Spase) for this combination or kfold and vocabulary_size
#corpus_mtrx_fname = crps_voc_path+'/kfold_CorpusMatrix_'+str(0)+str(5000)+'.h5'
#print "Loading pyTables TF EArray for CrossValidation for K-fold=", 0, " and Vocabulary size=", 5000
#Loading Coprus pyTables TF EArray for this combination or kfold and vocabulary_size
#h5f = tb.openFile(corpus_mtrx_fname, 'r+')
#corpus_mtrx = h5f.getNode('/',  'corpus_earray') #h5f.root.corpus_earray 

print corpus_mtrx[0].read()
doc_lens = np.sum(corpus_mtrx[0], axis=1)
print
print doc_lens
print
print "min:",doc_lens.max()," max:",doc_lens.min()

print "Normalizing"
max_col_arr = np.max(corpus_mtrx[0], axis=1)[:, np.newaxis]
for i, (row, max_val) in enumerate( zip(corpus_mtrx[0].iterrows(), max_col_arr) ):
    corpus_mtrx[0][i] = row / max_val

print corpus_mtrx[0]

doc_lens = np.sum(corpus_mtrx[0], axis=1)
print
print doc_lens
print
print "min:",doc_lens.max()," max:",doc_lens.min()


h5f.close()
    