# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 05:06:52 2018

@author: 张立辉
"""
import sys
import spacy
nlp = spacy.load('en_core_web_sm')

doc1 = nlp(sys.argv[1])
doc2 = nlp(sys.argv[2])
similarity = doc1.similarity(doc2)
print(similarity)