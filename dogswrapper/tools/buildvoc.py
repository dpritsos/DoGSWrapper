import sys
sys.path.append('../../../')
from html2vec.utils import tfdutils
from html2vec.base.io.basefilehandlers import file_list_frmpaths


def BuildingVocabulary(docs_model, docfiles_lst, docs_idxs=None, dims=None,
                       encoding='utf-8', error_handling='replace'):

    if type(docs_idxs) is list:

        # Getting the Document files selected by the list indeces 'docs_idxs'.
        docfiles_lst = list(docfiles_lst[docs_idxs])

    elif type(docs_idxs) is tuple and len(docs_idx) == 2:

        # Getting the Document files in range definde by the tuple  'docs_idxs'.
        docfiles_lst = list(docfiles_lst[range(docs_idxs[0], docs_idxs[1])])

    elif doc_idxs is None:

        # Getting the whole list of documents.
        docfiles_lst = list(docfiles_lst)

    else:
        raise Exception("Invalid input for argument: 'doc_idxs'")

    # Building the Vocabulary for the specific document model.
    print "Building the Vocabulary..."
    tf_vocab = docs_model.build_vocabulary(docfiles_lst, encoding, error_handling)

    # Keeping all the terms with same freq to the last feature of the requested size.
    real_dims = len(tr_vocab)
    if dims:
        tf_vocab = tfdutils.keep_atleast(tf_vocab, dims)

        # Getting the actual dimentions, i.e. features size, of the Vocabulary.
        real_dims = len(tf_vocab)

    # Creating the Terms-Index Vocabulary that is shorted by Frequency's descending order.
    tid_vocab = tfdutils.tf2tidx(resized_tf_vocab)

    return tid_vocab, tf_vocab, real_dims
