

import gensim
import numpy as np

dic = gensim.corpora.Dictionary([
    ['qw', 'e', 'f'], ['a', 'b', 'o']
])

print dict(zip(dic.token2id.values(), dic.token2id.keys()))

# numpy_matrix = np.random.randint(10, size=[5, 2])
# print "numpy", numpy_matrix

# corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
# print "corpus", corpus

# numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=5)
# print "numpy", numpy_matrix

train = gensim.matutils.Dense2Corpus(
    np.array(
        [[.1, .1, .0, .1, .0],
         [.2, .5, .5, .0, .0],
         [.0, .0, .1, .6, .1],
         [.2, .0, .0, .3, .5],
         [.1, .0, .0, .1, .1],
         [.1, .0, .0, .1, .1], # After this line a segmentation fault is cased
         [.1, .0, .0, .1, .1]]
    ).T
)

# print train

# mdl = gensim.models.LsiModel(
#     train,
#     num_topics=5
# )

mdl = gensim.models.doc2vec.Doc2Vec(train, size=3)

# print mdl.print_topics(2)
# gensim.models.LsiModel.print_topics()

# print dic.doc2bow('1 e a b'.split())
print gensim.matutils.corpus2dense(mdl[train], num_terms=5).T
print list(mdl[train])
# print mdl[train]
# print mdl.projection.u
# print mdl.projection.s



# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)



#lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI #transformation
#>>> corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: #bow->tfidf->fold-in-lsi
# model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
