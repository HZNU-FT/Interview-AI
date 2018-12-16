import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Model

import joblib as jl
import re

genre = 9
maxLen = 40

filepath = r'./input_text9/'
# filepath = r'./input_text3/'

epochs = 200

def rm(text):

	rm = re.compile(r'\n')
	return rm.sub('', text)

def read_files():

	all_labels = []
	all_texts = []
	file_list = []
	path = filepath

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

def preprocessing(train_texts, train_labels, test_texts, test_labels):

	tokenizer = Tokenizer(num_words=200)
	tokenizer.fit_on_texts(train_texts)
	x_train_seq = tokenizer.texts_to_sequences(train_texts)
	x_test_seq = tokenizer.texts_to_sequences(test_texts)
	x_train = sequence.pad_sequences(x_train_seq, maxlen=maxLen)
	x_test = sequence.pad_sequences(x_test_seq, maxlen=maxLen)
	y_train = np.array(train_labels)
	y_test = np.array(test_labels)

	with open("./models/tokenizer9", "wb") as handler:
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
	#output = Dense(units=3, activation='sigmoid')(output)

	model = Model([comment_seq], output)
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


train_texts, train_labels = read_files()
test_texts, test_labels = read_files()

x_train, y_train, x_test, y_test = preprocessing(train_texts, train_labels, test_texts, test_labels)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = text_cnn()

model.fit(x_train, y_train,
			validation_split=0.1,
			batch_size=128,
			epochs=epochs,
			shuffle=True)

scores = model.evaluate(x_test, y_test)
print(scores)

model.save('./models/text9_cnn_model.h5')


