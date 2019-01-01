import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import joblib as jl
import re

model = load_model('./models/text_cnn_model_demo.h5')

def preprocessing(question):

	with open("./models/tokenizer_demo", "rb") as handler:
		tokenizer = jl.load(handler)
	question = tokenizer.texts_to_sequences(question)
	question = sequence.pad_sequences(question, maxlen=150)

	return question

question = ['Bad']

question = preprocessing(question)

result = model.predict(question)

print(result)
print(np.argmax(result, axis=1))
