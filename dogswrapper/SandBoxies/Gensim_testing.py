

import gensim
import numpy as np

dic = gensim.corpora.Dictionary([['1', 'e'], ['a', 'b']])

print dic.__dict__

numpy_matrix = np.random.randint(10, size=[5, 2])
print numpy_matrix

corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
print corpus.__dict__

numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=5)
print numpy_matrix

mdl = gensim.models.LsiModel(
        gensim.matutils.Dense2Corpus(np.array([[1, 1], [2, 5]])),
        id2word={0: '1', 2: 'a', 3: 'b', 1: 'e'},
        num_topics=4
    )

print mdl.__dict__

# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)



#lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI #transformation
#>>> corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: #bow->tfidf->fold-in-lsi
# model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
