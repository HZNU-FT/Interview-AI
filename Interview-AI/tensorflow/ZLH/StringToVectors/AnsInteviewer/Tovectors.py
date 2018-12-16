# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:06:09 2018

@author: 张立辉
"""

import os

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('deerwester.dict')
corpus = corpora.MmCorpus('deerwester.mm')
   
tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]

vectors=[[vector[1] for vector in doc ] for doc in corpus_lsi]

f=open('Vectors.txt','x')

for vec in vectors:
    if len(vec)>0:
        f.write(str(vec))
        f.write("\n")

f.close()