import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Model

import joblib as jl

import tarfile
import numpy as np

import re

def text_cnn_sequential(maxlen=150, max_features=2000, embed_size=32):

	kernel_size = 3

	model = Sequential()
	model.add(Embedding(max_features, embed_size))
	model.add(Conv1D(filters=100, kernel_size=kernel_size, padding='valid', activation='relu', strides=1))

	model.add(MaxPooling1D())
	# model.add(GlobalMaxPooling1D())

	model.add(Dense(2, activation='softmax'))
	model.add(Dropout(0.2))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.summary()

	return model


def rm_tags(text):

	re_tag = re.compile(r'<[^>]+>')
	return re_tag.sub('', text)

def read_files(filetype):

	all_labels = [1]*12500 + [0]*12500
	all_texts = []
	file_list = []
	path = r'./aclImdb/'

	pos_path = path + filetype + '/pos/'
	for file in os.listdir(pos_path):
		file_list.append(pos_path + file)

	neg_path = path + filetype + '/neg/'
	for file in os.listdir(neg_path):
		file_list.append(neg_path + file)

	for file_name in file_list:
		with open(file_name, encoding='utf-8') as f:
			all_texts.append(rm_tags(" ".join(f.readlines())))

	return all_texts, all_labels


def preprocessing(train_texts, train_labels, test_texts, test_labels):

	tokenizer = Tokenizer(num_words=2000)
	tokenizer.fit_on_texts(train_texts)
	x_train_seq = tokenizer.texts_to_sequences(train_texts)
	x_test_seq = tokenizer.texts_to_sequences(test_texts)
	x_train = sequence.pad_sequences(x_train_seq, maxlen=150)
	x_test = sequence.pad_sequences(x_test_seq, maxlen=150)
	y_train = np.array(train_labels)
	y_test = np.array(test_labels)

	with open("./models/tokenizer_demo", "wb") as handler:
		jl.dump(tokenizer, handler)

	return x_train, y_train, x_test, y_test

def text_cnn(maxlen=150, max_features=2000, embed_size=32):
	
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

	output = Dense(units=2, activation='sigmoid')(output)

	model = Model([comment_seq], output)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


if not os.path.exists('./aclImdb'):
	tfile = tarfile.open(r'./aclImdb_v1.tar.gz', 'r:gz')  # r;gz是读取gzip压缩文件
	result = tfile.extractall('./')  # 解压缩文件到当前目录中

train_texts, train_labels = read_files('train')
test_texts, test_labels = read_files('test')
x_train, y_train, x_test, y_test = preprocessing(train_texts, train_labels, test_texts, test_labels)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = text_cnn()
# model = text_cnn_sequential()	

batch_size = 128
epochs = 10
model.fit(x_train, y_train,
			validation_split=0.1,
			batch_size=batch_size,
			epochs=epochs,
			shuffle=True)
scores = model.evaluate(x_test, y_test)
print(scores)

model.save('./models/text_cnn_model_demo.h5')
