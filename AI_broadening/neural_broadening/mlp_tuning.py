import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"

import hyperopt.early_stop
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pickle
from sklearn.metrics import mean_absolute_error
import keras
import numpy as np
import pandas as pd
import random
import scipy


minerg = 800
maxerg = 1500

print('Modules loaded, training started')

all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file
df = pd.read_csv('../AI_data/Fe56_200_to_1800_D1K_MT102.csv')
df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

test_temperatures = [1400]
validation_temperatures = []
nuclide = [26,56]






def build_model(hp):
	validation_temperatures = []
	while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
		choice = random.choice(all_temperatures)
		if choice not in validation_temperatures and choice not in test_temperatures:
			validation_temperatures.append(choice)

	training_temperatures = []
	for T in all_temperatures:
		if T not in test_temperatures and T not in validation_temperatures:
			training_temperatures.append(T)

	df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
	unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
	unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values

	test_dataframe = df[df['T'].isin(test_temperatures)]
	training_dataframe = df[df['T'].isin(training_temperatures)]
	logged_T_train = np.log(training_dataframe['T'].values)
	logged_ERG_train = np.log(training_dataframe['ERG'].values)
	X_train = np.array([scipy.stats.zscore(logged_ERG_train), scipy.stats.zscore(logged_T_train)])
	X_train = np.transpose(X_train)
	y_train_logged = np.array(np.log(training_dataframe['XS'].values))
	y_train = scipy.stats.zscore(y_train_logged)

	callback = keras.callbacks.EarlyStopping(monitor='val_loss',
											 # min_delta=0.005,
											 patience=10,
											 mode='min',
											 start_from_epoch=5,
											 restore_best_weights=True)