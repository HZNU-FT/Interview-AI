# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from gensim import corpora, models, similarities
from collections import defaultdict

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Model

import re
import joblib as jl

# 分类数目
genre = 0
# 最大长度
maxLen = 30
# 训练轮数
epochs = 500

# 读取文件所在路径
filepath = '../data/input_total/salary_question/'
savepath = './models/textCNN_model_salary_gensim.h5'

# 忽略的单词

# 不去掉停词
stoplist = set(''.split())

# 复杂版
# stoplist = set('please for a of the and to in are can who one i was youre should than our had an after now us we under dont two five about other do didnt so were will does I like you with your , if did at on as be from that minute whats im . ? / ( ) [ ] | this would this yourself tell why which any have it is could when or me what here how'.split())

# 简易版
# stoplist = set('please for a of is be the and to in are can about other do were I like you your ? , .'.split())

# 去除换行符
def rm(text):

	rm = re.compile(r'\n')
	return rm.sub('', text)

# 读文件
def readFile(path=filepath):

	all_texts = []
	all_labels = []
	file_list = []

	for file in os.listdir(path):
		file_list.append(path + file)

	counter_file = 0
	for file_name in file_list:
		with open(file_name, encoding='utf-8') as f:
			while 1:
				line = f.readline()
				if not line:
					counter_file += 1
					break
				all_texts.append(rm("".join(line)))
				all_labels.append(counter_file)

	genre = counter_file
	
	return all_texts, all_labels

# 预处理
def stringProcessing(documents):

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

	texts_concate = []
	#maxLenth = 0
	for sentence in texts:
		# if(len(sentence) > maxLenth):
		# 	maxLenth = len(sentence)
		texts_concate.append(" ".join(sentence))

	with open('./string/module/stoplist_salary', "wb") as handler:
		jl.dump(stoplist, handler)

	with open('./string/module/frequency_salary', "wb") as handler:
		jl.dump(frequency, handler)

	return texts_concate

def preprocessing(train_texts, train_labels, test_texts, test_labels):

	tokenizer = Tokenizer(num_words=200)
	tokenizer.fit_on_texts(train_texts)
	x_train_seq = tokenizer.texts_to_sequences(train_texts)
	x_test_seq = tokenizer.texts_to_sequences(test_texts)
	# print(x_train_seq)
	x_train = sequence.pad_sequences(x_train_seq, maxlen=maxLen)
	x_test = sequence.pad_sequences(x_test_seq, maxlen=maxLen)
	y_train = np.array(train_labels)
	y_test = np.array(test_labels)

	with open("./models/tokenizer_salary", "wb") as handler:
		jl.dump(tokenizer, handler)

	return x_train, y_train, x_test, y_test

def stringProcessing_gensim(documents):

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

	with open('./string/module/frequency_salary', "wb") as handler:
		jl.dump(frequency, handler)

	return texts

# 数组小数化
def gensim_corpus(dictionary, texts):

	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('./string/module/dictionary_salary.mm', corpus)

	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	# initialize an LSI transformation
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=maxLen) 
	corpus_lsi = lsi[corpus_tfidf]

	corpus_vectors = [[vector[1] for vector in doc ] for doc in corpus_lsi]

	return corpus_vectors

# 获取训练数组
def preprocessing_gensim(train_texts, train_labels, test_texts, test_labels):

	dictionary = corpora.Dictionary(train_texts)
	dictionary.save('./string/module/dictionary_salary.dict')

	x_train_seq = gensim_corpus(dictionary, train_texts)
	x_test_seq  = gensim_corpus(dictionary, test_texts)

	# 预处理小数，否则在pad过程中将会被忽略为0
	for index_i in range(len(x_train_seq)):
		for index_j in range(len(x_train_seq[index_i])):
			x_train_seq[index_i][index_j] *= 100
			x_train_seq[index_i][index_j] += 100
	for index_i in range(len(x_test_seq)):
		for index_j in range(len(x_test_seq[index_i])):
			x_test_seq[index_i][index_j] *= 100
			x_test_seq[index_i][index_j] += 100	 
	
	x_train = sequence.pad_sequences(x_train_seq, maxlen=maxLen)
	x_test = sequence.pad_sequences(x_test_seq, maxlen=maxLen)

	y_train = np.array(train_labels)
	y_test = np.array(test_labels)

	return x_train, y_train, x_test, y_test

def text_cnn(maxlen=maxLen, max_features=200, embed_size=32):

	comment_seq = Input(shape=[maxlen], name='x_seq')
	emb_comment = Embedding(max_features, embed_size)(comment_seq)

	convs = []
	filter_sizes = [2, 3, 4, 5]
	for fsz in filter_sizes:
		l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
		l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
		l_pool = Flatten()(l_pool)
		convs.append(l_pool)
	merge = concatenate(convs, axis=1)

	out = Dropout(0.5)(merge)
	output = Dense(32, activation='relu')(out)
	output = Dense(units=genre, activation='softmax')(output)
	#output = Dense(units=3, activation='sigmoid')(output)

	model = Model([comment_seq], output)
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

train_texts, train_lables = readFile()
test_texts, test_labels = readFile()

# 经过gensim的特征向量提取处理
train_texts = stringProcessing_gensim(train_texts)
test_texts = stringProcessing_gensim(test_texts)

x_train, y_train, x_test, y_test = preprocessing_gensim(train_texts, train_lables, test_texts, test_labels)

# 非gensim的特征向量提取处理
# train_texts = stringProcessing(train_texts)
# test_texts = stringProcessing(test_texts)

# x_train, y_train, x_test, y_test = preprocessing(train_texts, train_lables, test_texts, test_labels)

# print(x_train, x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = text_cnn()

model.fit(x_train, y_train,
			validation_split=0.1,
			batch_size=128,
			epochs=epochs,
			shuffle=True)

scores = model.evaluate(x_train, y_train)
print(scores)

# 保存模型
model.save(savepath)
