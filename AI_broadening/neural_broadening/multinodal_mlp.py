import pandas as pd
import scipy.interpolate
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from neural_broadening_functions import log_single_nuclide_data_maker

minerg = 800
maxerg = 1500

all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file

df = pd.read_csv('Fe56_200_to_1800_D1K_MT102.csv')

df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

test_temperatures = [1400]
validation_temperatures = []
nuclide = [26,56]

df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values


# interpolationFunction = scipy.interpolate.interp1d(unheated_energies, unheated_XS)

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