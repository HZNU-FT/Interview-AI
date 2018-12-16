# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 16:19:57 2018

@author: 张立辉
"""
# 打开文件 导入gensim
from gensim import corpora
f=open('questions.txt')
# 一口气读完文件
documents=f.readlines()

# 忽略的单词
stoplist = set('please for a of the and to in are can about other do were I like you your'.split())

# 遍历所有的词句，把不在忽略列表中的单词加入texts
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

# 创建字典并存储
dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict') 
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)  # store to disk, for later use

dic=dictionary.token2id

vec=[]
pos=0
for text in texts:
    vec.append([])
    for i in range(20):
        vec[pos].append(0)
    i=0
    for token in text:
        vec[pos][20-len(text)+i]=dic[token]+1
        i+=1
    pos+=1
print(vec)










