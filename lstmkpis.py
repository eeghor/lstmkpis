from keras.models import Sequential   	 # Sequential model is a linear stack of layers (layer instances)
from keras.layers.recurrent import LSTM  # Long Short-Term Memory layer
from keras.layers.core import Dense, Activation, Dropout  # Core layers

import pandas as pd

import numpy as np

# note: last value in sequence is the label
sequence_length = 100

model = Sequential()

"""
start adding layers; options for LSTM are these:

keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', 
					use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
					bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
					recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
					kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, 
					recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, 
					go_backwards=False, stateful=False, unroll=False)

"""
model.add(LSTM(input_shape=(sequence_length-1, 1), 
				units=32, # dimensionality of the output space
				return_sequences=True  # return the last output in the output sequence, or the full sequence
				))
model.add(Dropout(0.2))

model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.add(Activation('linear'))

"""
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, 
				weighted_metrics=None, target_tensors=None)
"""
model.compile(loss='mean_squared_error', optimizer='rmsprop')

n_grams = []

for ix in range(len(training_data)-sequence_length):
	n_grams.append(training_data[ix:ix+sequence_length])

n_grams_arr = normalize(np.array(n_grams))
np.random.shuffle(n_grams_arr)

# separate samples from labels
x = n_grams_arr[:, :-1]
labels = n_grams_arr[:, −1]

"""
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
		validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
		sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
"""
model.fit(x, labels, batch_size=50, nb_epochs=3, validation_split=0.05)

y_pred = model.predict(x_test)


