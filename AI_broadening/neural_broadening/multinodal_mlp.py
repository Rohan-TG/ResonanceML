import pandas as pd
# import scipy.interpolate
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm

# from neural_broadening_functions import log_single_nuclide_data_maker
nuclide = [26,56]
minerg = 800
maxerg = 1500

all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file
all_temperatures = all_temperatures[all_temperatures != 1250]
mean_alltemps = np.mean(all_temperatures)
std_alltemps = np.std(all_temperatures)
data_dir = '/Users/rntg/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT1K_samples/samples_csv'
# df = pd.read_csv('Fe56_200_to_1800_D1K_MT102.csv')


test_temperatures = [1400]
validation_temperatures = []
while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
	choice = random.choice(all_temperatures)
	if choice not in validation_temperatures and choice not in test_temperatures:
		validation_temperatures.append(choice)

training_temperatures = [T for T in all_temperatures if T not in validation_temperatures and T not in test_temperatures]
df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values










def dataMaker(temperatures):

	input_matrix = [] # contains the energy grid and temperatures
	labels_matrix = [] # contains the cross sections
	for T in tqdm.tqdm(temperatures, total = len(temperatures)):
		filestring = f'Fe56_T{int(T)}K.csv'
		df = pd.read_csv(f'{data_dir}/{filestring}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		unscaled_T_values = df['T'].values
		scaled_T_values = [(t - mean_alltemps) / std_alltemps for t in unscaled_T_values]

		unscaled_ERG = df['ERG'].values
		mean_ERG = np.mean(unscaled_ERG)
		std_ERG = np.std(unscaled_ERG)
		scaled_ERG = [(E - mean_ERG) / std_ERG for E in unscaled_ERG]

		input_submatrix = np.array([scaled_ERG, scaled_T_values]) # can add or remove ERG here to make energy an input parameter
		labelsubmatrix = np.array(df['XS'].values)

		input_matrix.append(input_submatrix)
		labels_matrix.append(labelsubmatrix)

	X = np.array(input_matrix)
	y = np.array(labels_matrix)

	flattened_y = y.flatten()
	meanXS = np.mean(flattened_y)
	stdXS = np.std(flattened_y)

	scaled_labels_matrix = []
	for set in labels_matrix:
		scaled_set = [(xs - meanXS) / stdXS for xs in set]
		scaled_labels_matrix.append(scaled_set)

	y = np.array(scaled_labels_matrix)

	return(X, y)






X_val, y_val = dataMaker(temperatures=validation_temperatures)
X_train, y_train = dataMaker(temperatures=training_temperatures)
X_test, y_test = dataMaker(temperatures=test_temperatures)




# for trainT in tqdm.tqdm(training_temperatures, total = len(training_temperatures)):
# 	filestring = f'Fe56_T{int(trainT)}K.csv'
# 	dftrain = pd.read_csv(f'{data_dir}/{filestring}')
# 	dftrain = dftrain[(dftrain['ERG'] < maxerg) & (dftrain['ERG'] > minerg)]
#
# 	unscaled_train_T = dftrain['T'].values
# 	scaled_train_T = [(t - mean_alltemps) / std_alltemps for t in unscaled_train_T]
#
# 	unscaled_ERG = dftrain['ERG'].values
# 	mean_ERG = np.mean(unscaled_ERG)
# 	std_ERG = np.std(unscaled_ERG)
# 	scaled_ERG_train = [(E - mean_ERG) / std_ERG for E in unscaled_ERG]
#
# 	Train_submatrix = np.array([scaled_ERG_train, scaled_train_T])  # can add or remove ERG here to make energy an input parameter
# 	labelsubmatrixtrain = np.array(dftrain['XS'].values)
#
# 	X_train.append(Train_submatrix)
# 	y_train.append(labelsubmatrixtrain)

# X_train = np.array(X_train)
# y_train = np.array(y_train)
#
# flattened_y_train = y_train.flatten()
# meantrainXS = np.mean(flattened_y_train)
# stdtrainXS = np.std(flattened_y_train)
#
# scaled_labels_validation_matrix = []
# for trainset in labels_validation_matrix:
# 	scaled_set = [(xs - meantrainXS) / stdtrainXS for xs in trainset]
# 	scaled_labels_validation_matrix.append(scaled_set)
#
# y_train = np.array(scaled_labels_validation_matrix)





# for testT in tqdm.tqdm(test_temperatures, total = len(test_temperatures)):
# 	filestring = f'Fe56_T{int(testT)}K.csv'
# 	dftrain = pd.read_csv(f'{data_dir}/{filestring}')
# 	dftrain = dftrain[(dftrain['ERG'] < maxerg) & (dftrain['ERG'] > minerg)]
#
# 	unscaled_train_T = dftrain['T'].values
# 	scaled_train_T = [(t - mean_alltemps) / std_alltemps for t in unscaled_train_T]
#
# 	unscaled_ERG = dftrain['ERG'].values
# 	mean_ERG = np.mean(unscaled_ERG)
# 	std_ERG = np.std(unscaled_ERG)
# 	scaled_ERG_train = [(E - mean_ERG) / std_ERG for E in unscaled_ERG]
#
# 	Train_submatrix = np.array([scaled_ERG_train, scaled_train_T])  # can add or remove ERG here to make energy an input parameter
# 	labelsubmatrixtrain = np.array(dftrain['XS'].values)
#
# 	X_train.append(Train_submatrix)
# 	y_train.append(labelsubmatrixtrain)

# X_train = np.array(X_train)
# y_train = np.array(y_train)
#
# flattened_y_train = y_train.flatten()
# meantrainXS = np.mean(flattened_y_train)
# stdtrainXS = np.std(flattened_y_train)
#
# scaled_labels_validation_matrix = []
# for trainset in labels_validation_matrix:
# 	scaled_set = [(xs - meantrainXS) / stdtrainXS for xs in trainset]
# 	scaled_labels_validation_matrix.append(scaled_set)
#
# y_train = np.array(scaled_labels_validation_matrix)








# callback = keras.callbacks.EarlyStopping(monitor='val_loss',
# 										 # min_delta=0.005,
# 										 patience=10,
# 										 mode='min',
# 										 start_from_epoch=5,
# 										 restore_best_weights=True)
#
# model = keras.Sequential()
# model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(y_test.shape[0], activation='linear'))
# model.compile(loss='mean_absolute_error', optimizer='adam')
#
# history = model.fit(X_train,
# 					y_train,
# 					epochs=30,
# 					batch_size=64,
# 					callbacks=callback,
# 					validation_data=(X_train, y_train),
# 					verbose=1)
#
#
# predictions = model.predict(X_train)
# predictions = predictions.ravel()