from __future__ import absolute_import,print_function,division
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import pandas as pd

corpus_raw = u""
nltk.download("punkt")
nltk.download("stopwords")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with codecs.open("/Users/karthik/Downloads/sentence2vec-gensim/model_data.txt", "r", "utf-8") as file:
        for line in file:
            corpus_raw += line                   #file.read()
raw_sentences = tokenizer.tokenize(corpus_raw)
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))
token_count = sum([len(sentence) for sentence in sentences])
print (format(token_count))
num_features = 300

#
# Minimum word count threshold.
min_word_count = 2

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1

java2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
java2vec.build_vocab(sentences)
java2vec.train(sentences,total_examples=java2vec.corpus_count,epochs=java2vec.iter)
if not os.path.exists("trained"):
    os.makedirs("trained")
java2vec.save(os.path.join("trained", "wikijava.bin"))
java2vec = w2v.Word2Vec.load(os.path.join("trained", "wikijava.bin"))
modj = w2v.Word2Vec.load(os.path.join("trained", "wikijava.bin"))
print("it worked!!")
print(java2vec.most_similar("spring"))
print(java2vec.similarity('Spring','Java'))
