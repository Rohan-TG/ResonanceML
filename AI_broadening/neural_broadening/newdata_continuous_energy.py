import os
import time

import tqdm

# Set environment variables to limit TensorFlow to 30 CPU cores
os.environ["OMP_NUM_THREADS"] = "30"
os.environ["TF_NUM_INTRAOP_THREADS"] = "30"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # Adjust inter-op parallelism if needed
os.environ["OPENBLAS_NUM_THREADS"] = "30"
os.environ["MKL_NUM_THREADS"] = "30"
#
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
import tensorflow as tf
import scipy.stats
from sklearn.metrics import mean_absolute_error
import keras
import matplotlib.pyplot as plt
import numpy as np
from neural_broadening_functions import log_single_nuclide_data_maker
import pandas as pd
import random
import periodictable
import datetime


def get_datetime_string():
    return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")



minerg = 1000
maxerg = 1200

mintemp = 800
maxtemp = 1200
all_temperatures = np.arange(mintemp, maxtemp, 0.1) # all temperatures in the data file



data_dir = '/home/rnt26/NJOY/data/Fe56_JEFF/CSVs'

test_temperatures = [1000]
nuclide = [26,56]




df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values

validation_temperatures = []
while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
	choice = random.choice(all_temperatures)
	if choice not in validation_temperatures and choice not in test_temperatures:
		validation_temperatures.append(choice)

training_temperatures = []
for T in all_temperatures:
	if T not in test_temperatures and T not in validation_temperatures:
		training_temperatures.append(T)




# stat stuff for temperatures
logged_temperatures = np.log10(np.array(all_temperatures).astype(np.float128))
mean_alltemps = np.mean(logged_temperatures.astype(np.float128), dtype=np.float128)
std_alltemps = np.std(logged_temperatures.astype(np.float128), dtype= np.float128)

filenames = os.listdir(data_dir)
exclusions = []

T_train = []
XS_train = []
ERG_train = []

for train_temperature in tqdm.tqdm(training_temperatures, total = len(training_temperatures)):
	if round(float(train_temperature), 1) not in exclusions:
		roundedtt = str(round(train_temperature, 1))
		filename = f'Fe56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]



		logged_T_values = np.log10(df['T'].values)
		scaled_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_T_values]


		T_train.append(scaled_T_values)  # can add or remove ERG here to make energy an input parameter
		XS_train.append(np.log10(df['XS'].values))
		ERG_train.append(df['ERG'].values)

flat_T_Train = [item for sublist in T_train for item in sublist]
flat_ERG_train = [item for sublist in ERG_train for item in sublist]
flat_XS_train = [item for sublist in XS_train for item in sublist]

logged_erg_train = np.log10(flat_ERG_train)
erg_train_mean = np.mean(logged_erg_train)
erg_train_std = np.std(logged_erg_train)
scaled_erg_train = [(erg - erg_train_mean) / erg_train_std for erg in logged_erg_train]


logged_xs_train = np.log10(flat_XS_train)
xs_train_mean = np.mean(logged_xs_train)
xs_train_std = np.std(logged_xs_train)
scaled_xs_train = [(xs - xs_train_mean) / xs_train_std for xs in logged_xs_train]


X_train = np.array([scaled_erg_train, flat_T_Train])
y_train = scaled_xs_train




# callback = keras.callbacks.EarlyStopping(monitor='val_loss',
# 										 # min_delta=0.005,
# 										 patience=5,
# 										 mode='min',
# 										 start_from_epoch=5,
# 										 restore_best_weights=True)
#
# model = keras.Sequential()
# model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(50, activation='relu'))
# model.add(keras.layers.Dense(20, activation='relu'))
# model.add(keras.layers.Dense(10,activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
# model.compile(loss='mae', optimizer='adam')
#
# history = model.fit(X_train,
# 					y_train,
# 					epochs=100,
# 					batch_size=32,
# 					callbacks=callback,
# 					validation_data=(X_train, y_train),
# 					verbose=1)
#
# predictions = model.predict(X_test)
# predictions = predictions.ravel()
#
#
#
# scaled_energies = []
# for pair in X_test:
# 	scaled_energies.append(pair[0])
#
#
#
#
# # logged_ERG_test = np.log(ERG_test)
# # logged_y_test = np.log(y_test)
#
# rescaled_energies = np.array(scaled_energies) * np.std(logged_ERG_test) + np.mean(logged_ERG_test)
# rescaled_energies = np.e ** rescaled_energies
#
# rescaled_predictions = np.array(predictions) * np.std(logged_y_test) + np.mean(logged_y_test)
# rescaled_predictions = np.e ** rescaled_predictions
#
# rescaled_test_xs = np.array(y_test) #* np.std(df['XS'].values) + np.mean(df['XS'].values)
#
#
#
#
#
# def bounds(lower_bound, upper_bound, scalex='log', scaley='log'):
# 	unheated_energies_limited = []
# 	unheated_XS_limited = []
# 	for x, h in zip(unheated_energies, unheated_XS):
# 		if x <= upper_bound and x >= lower_bound:
# 			unheated_energies_limited.append(x)
# 			unheated_XS_limited.append(h)
#
# 	test_energies_limited = []
# 	predictions_limited = []
# 	test_XS_limited = []
# 	rescaled_test_XS = test_dataframe['XS'].values
# 	for o, p, qx in zip(rescaled_energies, rescaled_predictions, rescaled_test_XS):
# 		if o <= upper_bound and o >= lower_bound:
# 			test_energies_limited.append(o)
# 			predictions_limited.append(p)
# 			test_XS_limited.append(qx)
#
# 	plt.figure()
# 	# plt.plot(unheated_energies_limited, unheated_XS_limited, label = '0 K JEFF-3.3')
# 	plt.grid()
# 	plt.plot(test_energies_limited, predictions_limited, label='Predictions', color='red')
# 	plt.xlabel('Energy / eV')
# 	plt.ylabel('$\sigma_{n,\gamma} / b$')
# 	plt.plot(test_energies_limited, test_XS_limited, '--', label=f'{test_temperatures[0]} K JEFF-3.3', color='lightgreen',
# 			 alpha=0.7)
# 	plt.legend()
# 	plt.xscale('log')
# 	plt.yscale('log')
# 	plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
# 	# if scaley == 'log':
# 	# 	plt.yscale('log')
# 	# else:
# 	# 	plt.yscale('linear')
# 	# if scalex ==' log':
# 	# 	plt.xscale('log')
# 	# else:
# 	# 	plt.xscale('linear')
# 	plt.show()
#
#
# 	relativeError = []
# 	percentageError = []
# 	for p, xs in zip(predictions_limited, test_XS_limited):
# 		relativeError.append(abs((p-xs)/xs))
# 		percentageError.append((p/xs * 100) - 100)
#
#
#
# 	plt.figure()
# 	plt.plot(test_energies_limited, relativeError, label = 'Error')
# 	plt.xlabel('Energy / eV')
# 	plt.ylabel('Relative error')
# 	plt.xscale('log')
# 	plt.legend()
# 	plt.yscale('log')
# 	plt.grid()
# 	plt.show()
#
# 	plt.figure()
# 	plt.plot(test_energies_limited, percentageError, label='Error')
# 	plt.xlabel('Energy / eV')
# 	plt.ylabel('% Error')
# 	plt.grid()
# 	# plt.savefig(f'/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/neural_broadening/miscplots/mlpeerrors_{timestring}.png', dpi = 300)
# 	plt.show()
#
# 	plt.figure()
# 	plt.hist(percentageError, bins=50)
# 	plt.ylabel('Frequency')
# 	plt.xlabel('% Error')
# 	plt.grid()
# 	plt.show()
#
# 	countoverthreshold = 0
# 	for XX in percentageError:
# 		if abs(XX) >= 0.1:
# 			countoverthreshold += 1
#
# 	percentageOverThreshold = (countoverthreshold / (len(percentageError))) * 100
#
# 	print(f'Max error: {np.max(abs(np.array(percentageError)))}')
# 	print(f'Mean error: {np.mean(abs(np.array(percentageError)))}')
# 	print(f'{percentageOverThreshold} % of points over limit of 0.1 % error')
#
#
#
#
#
# test_xs = test_dataframe['XS'].values
#
# timestring = get_datetime_string()
#
#
# plt.figure()
# # plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
# plt.plot(ERG_test, test_xs, '--', label = 'JEFF-3.3 1,800 K')
# plt.plot(rescaled_energies, rescaled_predictions, label = 'Predictions', color = 'red')
# plt.legend()
# plt.grid()
# plt.xlabel('Energy / eV')
# plt.ylabel('$\sigma_{n,\gamma} / b$')
# plt.xscale('log')
# plt.yscale('log')
# # plt.savefig(f'/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/neural_broadening/miscplots/mlpplot-{timestring}_pres.png', dpi = 300)
# plt.show()
#
# plt.figure()
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.plot(history.history['loss'], label='Training Loss')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.show()
#
# MAE = mean_absolute_error(rescaled_predictions, test_xs)
# bounds(minerg, maxerg)
