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


def preprocessing(question):

	with open("./models/tokenizer9", "rb") as handler:
		tokenizer = jl.load(handler)

	#x_train_seq = tokenizer.texts_to_sequences(train_texts)
	question = tokenizer.texts_to_sequences(question)
	#x_train = sequence.pad_sequences(x_train_seq, maxlen=30)
	question = sequence.pad_sequences(question, maxlen=maxLen)

	return question


question_origin = [	'Would you be willing to take less money ?', 
					'What are your greatest strengths ?', 
					'What other types of jobs or companies are you considering ?', 
					'Give me a summary of your current job description .', 
					'What are your short-term objectives ?']

question_origin = preprocessing(question_origin)

result_origin = model.predict(question_origin)
result_array_origin = np.argmax(result_origin, axis=1)

print(result_origin)
print(result_array_origin)


question_new = ['Can you introduce yourself ?', 
				'Please give me a summary of your work description .', 
				'What types of jobs are you interested in ?', 
				'Tell me about your greatest strengths .', 
				'Which course is your favorite at school ?']

question_new = preprocessing(question_new)

result_new = model.predict(question_new)
result_array_new = np.argmax(result_new, axis=1)

print(result_new)
print(result_array_new)
