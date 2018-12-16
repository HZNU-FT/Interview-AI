# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 05:44:23 2018

@author: 张立辉
"""

from gensim import corpora

f=open('anwers.txt');
documents=f.readlines();

texts=[text.split() for text in documents]

dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict') 
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)  # store to disk, for later use

f.close()