import pandas as pd
# import scipy.interpolate
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import random
# from neural_broadening_functions import log_single_nuclide_data_maker
nuclide = [26,56]
minerg = 800
maxerg = 1500

all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file
data_dir = '/Users/rntg/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT1K_samples/samples_csv'
# df = pd.read_csv('Fe56_200_to_1800_D1K_MT102.csv')


test_temperatures = [1400]
validation_temperatures = []
while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
	choice = random.choice(all_temperatures)
	if choice not in validation_temperatures and choice not in test_temperatures:
		validation_temperatures.append(choice)


df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values

validation_matrix = []
for valT in validation_temperatures:
	filestring = f'Fe56_T{int(valT)}K.csv'
	dfval = pd.read_csv(f'{data_dir}/{filestring}')
	submatrix = np.array(dfval)
	validation_matrix.append(submatrix)
	break
validation_matrix = np.array(validation_matrix)
# df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]


training_matrix = []
test_matrix = []

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=10,
										 mode='min',
										 start_from_epoch=5,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(y_test.shape[0], activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=30,
					batch_size=64,
					callbacks=callback,
					validation_data=(X_train, y_train),
					verbose=1)


predictions = model.predict(X_train)
predictions = predictions.ravel()