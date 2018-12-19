# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from gensim import corpora
from collections import defaultdict

import re
import joblib as jl

# 读取文件所在路径
filepath = '../../data/input_total/question/'

# 忽略的单词
stoplist = set('please for a of the and to in are can who one i was youre should than our had an after now us we under dont two five about other do didnt so were will does I like you with your , if did at on as be from that minute whats im . ? / ( ) [ ] | this would this yourself tell why which any have it is could when or me what here how'.split())

# 去除换行符
def rm(text):

    rm = re.compile(r'\n')
    return rm.sub('', text)

# 读文件
def readFile(path):

    documents = []
    file_list = []

    for file in os.listdir(path):
        file_list.append(path + file)

    for file_name in file_list:
        with open(file_name, encoding='utf-8') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                documents.append(rm("".join(line)))

    return documents

# 预处理
def preprocessing(documents):

    frequency = defaultdict(int)

    # 遍历所有的词句，把不在忽略列表中的单词加入texts
    texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

    # 记录词出现频率
    for text in texts:
        for token in text:
            frequency[token] += 1

    # 将出现次数大于一的保留
    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    return texts, frequency


documents = readFile(filepath)

texts, frequency = preprocessing(documents)

# 保存停词列表与词频字典
# with open("./module/stoplist", "wb") as handler:
#     jl.dump(stoplist, handler)
# with open("./module/frequency", "wb") as handler:
#     jl.dump(frequency, handler)


dictionary = corpora.Dictionary(texts)
dictionary.save('./module/dictionary.dict') 

# 将词向量的每项整数小数化
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./module/dictionary.mm', corpus)

print("Successfully dumped stoplist, frequency and gensim dictionary.")

# dic=dictionary.token2id

# vec=[]
# pos=0
# for text in texts:
#     vec.append([])
#     for i in range(20):
#         vec[pos].append(0)
#     i=0
#     for token in text:
#         vec[pos][20-len(text)+i]=dic[token]+1
#         i+=1
#     pos+=1
# print(vec)
