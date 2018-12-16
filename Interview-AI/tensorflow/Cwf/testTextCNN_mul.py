import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import joblib as jl
import re

model = load_model('./models/text9_cnn_model.h5')

genre = 9
maxLen = 40

filepath = r'./input_text9/'
# filepath = r'./input_text3/'

def rm(text):

	rm = re.compile(r'\n')
	return rm.sub('', text)

def readFile(path):
	
	questions = []
	labels = []
	file_list = []

	for file in os.listdir(path):
		file_list.append(path + file)

	counter_question = 0
	counter_file = 0
	for file_name in file_list:
		with open(file_name, encoding='utf-8') as f:
			while 1:
				line = f.readline()
				if not line:
					counter_file += 1
					break;
				questions.append(rm("".join(line)))
				labels.append(counter_file)
				counter_question += 1

	return questions, labels, counter_question

def preprocessing(question):

	with open("./models/tokenizer9", "rb") as handler:
		tokenizer = jl.load(handler)

	#x_train_seq = tokenizer.texts_to_sequences(train_texts)
	question = tokenizer.texts_to_sequences(question)
	#x_train = sequence.pad_sequences(x_train_seq, maxlen=30)
	question = sequence.pad_sequences(question, maxlen=maxLen)

	return question


questions, labels, num_questions = readFile(r'./input_text9/')

questions_array = preprocessing(questions)

result = model.predict(questions_array)

result_array = np.argmax(result, axis=1)

# 输出所有预测结果
# for index in range(num_questions):
# 	print(questions[index], result_array[index])

# 错误语句输出
counter_failed = 0
for index in range(num_questions):
	if labels[index] != result_array[index]:
		counter_failed += 1
		print(questions[index], ', 预测分类为: ', result_array[index], ', 正确分类为: ', labels[index])

print('总共测试 ', num_questions, ' 个样本, ', counter_failed, ' 个样本预测失败, 正确率为: ', (num_questions - counter_failed) / num_questions)
