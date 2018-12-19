# 对经过去停词、频率筛选后的训练模型进行测试
# 集合了批量测试和自定义测试两种功能，注释部分代码即可
# 暂未加入gensim预处理函数

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import joblib as jl
import re

filepath = '../data/input_total/question/'
model = load_model('./models/textCNN_model_gensim.h5')

genre = 9
maxLen = 20

def rm(text):

	rm = re.compile(r'\n')
	return rm.sub('', text)

def readFile(path=filepath):
	
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

# 去停词与出现频率低的词
def stringProcessing(questions):

	with open('./string/module/stoplist', "rb") as handler:
		stoplist = jl.load(handler)
	with open('./string/module/frequency', "rb") as handler:
		frequency = jl.load(handler)

	questions = [[word for word in question.lower().split() if word not in stoplist]
			for question in questions]

	# 将出现次数大于一的保留
	questions = [[token for token in question if frequency[token] > 1]
			 for question in questions]

	question_concate = []
	for question in questions:
		question_concate.append(" ".join(question))

	return question_concate

# 字典预处理，转化为矩阵
def preprocessing(question):

	with open('./models/tokenizer_gensim', "rb") as handler:
		tokenizer = jl.load(handler)

	question = tokenizer.texts_to_sequences(question)
	question = sequence.pad_sequences(question, maxlen=maxLen)

	return question

# 输出批量测试结果``````````````````````````

# questions, labels, num_questions = readFile()

# questions = stringProcessing(questions)
# questions_array = preprocessing(questions)

# result = model.predict(questions_array)
# result_array = np.argmax(result, axis=1)

# # 输出所有预测结果
# # for index in range(num_questions):
# # 	print(questions[index], result_array[index])

# # 错误语句输出
# counter_failed = 0
# for index in range(num_questions):
# 	if labels[index] != result_array[index]:
# 		counter_failed += 1
# 		print(questions[index], ', 预测分类为: ', result_array[index], ', 正确分类为: ', labels[index])

# print('总共测试 ', num_questions, ' 个样本, ', counter_failed, ' 个样本预测失败, 正确率为: ', (num_questions - counter_failed) / num_questions)

# 自定义问题的分类测试``````````````````````````

# 正确结果是 0 2 7 3 5 4 8
# 注：分类9的问题经常会被分错类

questions = ['Could you please introduce yourself ?', 
				'Do you consider yourself successful ?', 
				'What do you enjoy most about what you do now ?', 
				'Who was your best boss and who was the worst ?', 
				'What about your salary benefits ?',
				'What was your biggest accomplishment on the job ? ',
				'What type of work environment do you prefer ?'
				]

questions = stringProcessing(questions)
question_array = preprocessing(questions)

result = model.predict(question_array)
result_array = np.argmax(result, axis=1)

print(result)
print(result_array)
