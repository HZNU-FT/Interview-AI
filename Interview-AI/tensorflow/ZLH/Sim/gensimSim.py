# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:27:33 2018

@author: 张立辉
"""
import sys

anId=sys.argv[1]
doc=sys.argv[2]

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('py/Similaritie/deerwester.dict')
corpus = corpora.MmCorpus('py/Similaritie/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=30)

vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space

index = similarities.MatrixSimilarity(lsi[corpus])

sims = index[vec_lsi]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
re=0
for sim in sims:
    if sim[0]==int(anId):
        re=sim[1]
print(re)
