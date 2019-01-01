import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import joblib as jl
import re


def rm(text):

	rm = re.compile(r'\n')
	return rm.sub('', text)

def read_files():

	all_labels = []
	all_texts = []
	file_list = []
	path = r'./input_text/'

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

train_texts, train_labels = read_files()

print(train_texts, train_labels)
