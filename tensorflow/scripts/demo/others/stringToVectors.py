# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 09:00:18 2018

@author: 张立辉
"""

# 用于保存成前方非零的向量
import os

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('./module/dictionary.dict')
corpus = corpora.MmCorpus('./module/dictionary.mm')
   
tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]

vectors=[[vector[1] for vector in doc ] for doc in corpus_lsi]

print(len(vectors))
