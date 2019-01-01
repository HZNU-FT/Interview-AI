# -*- coding: utf-8 -*-

# 完成版训练脚本

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

# 预处理相关
maxLen = 50
stoplist = set(''.split())

# 训练相关
genre = 9
epochs = 200

# 文件操作相关

# 面试者问题分类器相关参数
# path_input_train = '../data/input_total/salary_question/'
# path_input_test = '../data/input_total/salary_question/'

# path_stoplist = './modules/stoplist_salary'
# path_frequency = './modules/frequency_salary'
# path_tokenizer = './modules/tokenizer_salary'

# path_model = './models/textCNN_model_salary.h5'

# 面试官问题分类器相关参数
path_input_train = '../data/input_total/question/'
path_input_test = '../data/input_total/question/'

path_stoplist = './modules/stoplist'
path_frequency = './modules/frequency'
path_tokenizer = './modules/tokenizer'

path_model = './models/textCNN_model.h5'


# 去除换行符
def rm(text):

	rm = re.compile(r'\n')
	return rm.sub('', text)

# 读文件
def readFile(path):

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
	for sentence in texts:
		texts_concate.append(" ".join(sentence))

	with open(path_stoplist, "wb") as handler:
		jl.dump(stoplist, handler)

	with open(path_frequency, "wb") as handler:
		jl.dump(frequency, handler)

	return texts_concate

def preprocessing(train_texts, train_labels, test_texts, test_labels):

	tokenizer = Tokenizer(num_words=200)
	tokenizer.fit_on_texts(train_texts)
	x_train_seq = tokenizer.texts_to_sequences(train_texts)
	x_test_seq = tokenizer.texts_to_sequences(test_texts)
	x_train = sequence.pad_sequences(x_train_seq, maxlen=maxLen)
	x_test = sequence.pad_sequences(x_test_seq, maxlen=maxLen)
	y_train = np.array(train_labels)
	y_test = np.array(test_labels)

	with open(path_tokenizer, "wb") as handler:
		jl.dump(tokenizer, handler)

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
	
	model = Model([comment_seq], output)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

# ````````````````````````````````````````````````

# 读取
train_texts, train_lables = readFile(path_input_train)
test_texts, test_labels = readFile(path_input_test)

# 预处理
train_texts = stringProcessing(train_texts)
test_texts = stringProcessing(test_texts)
x_train, y_train, x_test, y_test = preprocessing(train_texts, train_lables, test_texts, test_labels)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 训练
model = text_cnn()
model.fit(x_train, y_train,
			validation_split=0.1,
			batch_size=128,
			epochs=epochs,
			shuffle=True)

# 测试
scores = model.evaluate(x_train, y_train)
print(scores)

# 保存模型
model.save(path_model)
