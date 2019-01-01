import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import joblib as jl
import re

genre = 4
maxLen = 30

# path_input = '../data/input_total/question/'

# path_stoplist = './modules/stoplist'
# path_frequency = './modules/frequency'
# path_tokenizer = './modules/tokenizer'

# path_model = './models/textCNN_model.h5'


path_input = '../data/input_total/salary_question/'

path_stoplist = './modules/stoplist_salary'
path_frequency = './modules/frequency_salary'
path_tokenizer = './modules/tokenizer_salary'

path_model = './models/textCNN_model_salary.h5'


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

# 去停词与出现频率低的词
def stringProcessing(questions):

	with open(path_stoplist, "rb") as handler:
		stoplist = jl.load(handler)
	with open(path_frequency, "rb") as handler:
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

	with open(path_tokenizer, "rb") as handler:
		tokenizer = jl.load(handler)

	question = tokenizer.texts_to_sequences(question)
	question = sequence.pad_sequences(question, maxlen=maxLen)

	return question


model = load_model(path_model)


# 输出批量测试结果``````````````````````````

# questions, labels, num_questions = readFile(path_input)

# questions = stringProcessing(questions)
# questions_array = preprocessing(questions)

# result = model.predict(questions_array)
# result_array = np.argmax(result, axis=1)

# # 输出所有预测结果
# for index in range(num_questions):
# 	print(questions[index], result_array[index])

# # 错误语句输出
# counter_failed = 0
# for index in range(num_questions):
# 	if labels[index] != result_array[index]:
# 		counter_failed += 1
# 		print(questions[index], ', 预测分类为: ', result_array[index], ', 正确分类为: ', labels[index])

# print('总共测试 ', num_questions, ' 个样本, ', counter_failed, ' 个样本预测失败, 正确率为: ', (num_questions - counter_failed) / num_questions)

# 自定义问题的分类测试``````````````````````````

# 0 1 2 3 4 5 6 7 8 8 8 8
questions = [	'Can I know something about you ?', 
				'Have you worked with someone you didnt like? schooljobsummary If so , how did you handle it ?', 
				'How would your colleagues describe your personality ?', 
				'Please give me your definition of Standard dialect .', 
				'What is your management style ?',
				'What were they paying you , If you dont mind my asking ? ',
				'Why are you applying for this position ?',
				'What is your long term career aspiration ?',
				'Do you work best by yourself or as part of a team ?',
				'What type of work environment do you prefer ?',
				'What provide you with a sense of accomplishment .',
				'What was your relationship with co-workers ?'
			]

# 0 1 2 3 3 3 3
# questions = ['What salary are you seeking ?',
# 			'I want to have another try .',
# 			'can i live in companys domitory ?',
# 			'Do you have any requitements on the workers compensation ?',
# 			'how many days off can i have in a month',
# 			'How many day offs do you want',
# 			'How about your retirement pension'
# 			]

questions = stringProcessing(questions)
question_array = preprocessing(questions)

result = model.predict(question_array)
result_array = np.argmax(result, axis=1)

print(result)
print(result_array)
