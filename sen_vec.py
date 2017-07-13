#####################################################################
# Implementation of Simple But Tough to beat sentence embeddings
#
# Author: Prudhvi
#
#####################################################################

from __future__ import division
import gensim
import numpy as np
from sklearn.decomposition import PCA
import gensim.models.word2vec
from collections import Counter
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
logging.basicConfig(level=logging.INFO)
starttime = time.time()
logger = logging.getLogger(__name__)
def gensim_load_vec(path):
    logger.info("word2vec loading......") 
    #use gensim_emb.wv.index2word if used this way to load vectors
    gensim_emb = gensim.models.word2vec.Word2Vec.load(path)
    #gensim_emb =  gensim.models.KeyedVectors.load_word2vec_format(path, binary=True,limit=500000)
    vocab = gensim_emb.wv.index2word
    vec = gensim_emb.wv.syn0
    shape = gensim_emb.wv.syn0.shape
    return gensim_emb, vec, shape, vocab
 
def map_word_frequency(document):
    return Counter(itertools.chain(*document))
    
def sentence2vec(tokenised_sentence_list, embedding_size, word_emb_model, a = 1e-3):
    word_counts = map_word_frequency(tokenised_sentence_list)
    sentence_set=[]
    for sentence in tokenised_sentence_list:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF
        try:
            vs = np.add(vs, np.multiply(a_value, word_emb_model[word])) # vs += sif * word_vector
        except Exception as e:
            print("invalid sentences")
        vs = np.divide(vs, sentence_length) # weighted average
        sentence_set.append(vs)	
    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.explained_variance_ratio_  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT
 
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below
 
    # resulting sentence vectors, vs = vs - u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))
 	
    return sentence_vecs

glove, glove_vec, glove_shape, glove_vocab = gensim_load_vec('/Users/karthik/Downloads/sentence2vec-gensim/trained/wikijava.bin')
tweets = ["This is Java","Java is language"]
if(tweets[0]==tweets[1]):
	print("same sentences are given")
else:
	tweets = [tweet.split() for tweet in tweets]
	embedding_size = glove_shape[1]
	logger.info("coverting sentences to vectors....") 
	sent_emb = sentence2vec(tweets, embedding_size, glove)
	print("semantic similarity")
	print(float(cosine_similarity([sent_emb[0]],[sent_emb[1]])))
print("Execution time is %f"%(time.time()-starttime))

