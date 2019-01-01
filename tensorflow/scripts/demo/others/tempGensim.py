# -*- coding: utf-8 -*-
from gensim import corpora

def rm(texts):

    for d in documents:
        d = d[0:len(d) - 2]

    return documents

def readFile(path):

    f = open(path)
    d = f.readlines()

    return d

def preprocessing(documents):


    documents = rm(documents)
    stoplist = set('please for a of the and to in are can about other do were'.split())

    texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

    return texts


def iterAddIndex(dictionary, maxLen):

    dic = dictionary.token2id

    # 将字典中的编号+1
    data = []
    i = 0
    for iter in dic:
        dic[iter] += 1
    # 创建句子的特征向量,将单词替换成字典中的id
    for line in texts:

        data.append([])
        for j in range(maxLen):
            data[i].append(0)

        length = len(line)
        k = 0

        for word in line:
            pos = dic[word]
            data[i][maxLen - length + k] = pos
            k += 1

        i += 1

    dictionary = dic.id2token
    
    return dictionary

documents = readFile('./input_text/questions.txt')

texts = preprocessing(documents)

# 获取最大长度
maxLen = 0
for sentence in texts:
    if len(sentence) > maxLen:
        maxLen = len(sentence)

# 创建字典并存储
dictionary = corpora.Dictionary(texts)
dictionary = iterAddIndex(dictionary, maxLen)

print(dictionary)

dictionary.save('./models/cnn.dict') 

'''
读取字典
dictionary = corpora.Dictionary.load('deerwester.dict')
'''
