import pandas as pd
# import scipy.interpolate
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm
import periodictable
import datetime

# from neural_broadening_functions import log_single_nuclide_data_maker
nuclide = [26,56]
minerg = 900
maxerg = 1300

plotdir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/neural_broadening/multinodalplots'

all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file
all_temperatures = all_temperatures[all_temperatures != 1250]
log_alltemps = np.log10(all_temperatures)
mean_alltemps = np.mean(log_alltemps)
std_alltemps = np.std(log_alltemps)
data_dir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT1K_samples/samples_csv'


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



def get_datetime_string():
	return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
timestring = get_datetime_string()










def dataMaker(temperatures):

	input_matrix = [] # contains the energy grid and temperatures
	labels_matrix = [] # contains the cross sections
	for T in tqdm.tqdm(temperatures, total = len(temperatures)):
		filestring = f'Fe56_T{int(T)}K.csv'
		df = pd.read_csv(f'{data_dir}/{filestring}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		unscaled_T_values = df['T'].values
		scaled_T_values = [(t - mean_alltemps) / std_alltemps for t in unscaled_T_values]

		# unscaled_ERG = df['ERG'].values
		# mean_ERG = np.mean(unscaled_ERG)
		# std_ERG = np.std(unscaled_ERG)
		# scaled_ERG = [(E - mean_ERG) / std_ERG for E in unscaled_ERG]

		input_submatrix = np.array(scaled_T_values) # can add or remove ERG here to make energy an input parameter
		labelsubmatrix = np.array(np.log10(df['XS'].values))

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
X_trainoriginal, y_trainoriginal = dataMaker(temperatures=training_temperatures)
X_test, y_test = dataMaker(temperatures=test_temperatures)


X_train = np.tile(X_trainoriginal, (3,1))
y_train = np.tile(y_trainoriginal, (3,1))




callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=10,
										 mode='min',
										 start_from_epoch=5,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(X_train.shape[1], input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(X_train.shape[1]))
model.add(keras.layers.LeakyReLU(alpha=0.05))
model.add(keras.layers.Dense(X_train.shape[1]))
model.add(keras.layers.LeakyReLU(alpha=0.05))
model.add(keras.layers.Dense(X_train.shape[1]))
model.add(keras.layers.LeakyReLU(alpha=0.05))
# model.add(keras.layers.Dense(X_train.shape[1], activation='relu'))
# model.add(keras.layers.Dense(X_train.shape[1], activation='relu'))
model.add(keras.layers.Dense(X_train.shape[1]))
model.add(keras.layers.LeakyReLU(alpha=0.05))
# model.add(keras.layers.LeakyReLU(alpha=0.05))
model.add(keras.layers.Dense(y_test.shape[1], activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=500,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_train, y_train),
					verbose=1)


predictions = model.predict(X_test)
predictions = predictions.ravel()

teststring = f'Fe56_T{int(test_temperatures[0])}K.csv'
dftest = pd.read_csv(f'{data_dir}/{teststring}')
dftest = dftest[(dftest['ERG'] < maxerg) & (dftest['ERG'] > minerg)]

testxs = dftest['XS'].values
meantestxs = np.mean(np.log10(testxs))
stdtestxs = np.std(np.log10(testxs))

rescaled_predictions = [p * stdtestxs + meantestxs for p in predictions]
energies = dftest['ERG'].values

rescaled_predictions = [10 ** P for P in rescaled_predictions]


def bounds(lower_bound, upper_bound, scalex='log', scaley='log'):
	unheated_energies_limited = []
	unheated_XS_limited = []
	for x, h in zip(unheated_energies, unheated_XS):
		if x <= upper_bound and x >= lower_bound:
			unheated_energies_limited.append(x)
			unheated_XS_limited.append(h)

	test_energies_limited = []
	predictions_limited = []
	test_XS_limited = []
	for o, p, qx in zip(energies, rescaled_predictions, testxs):
		if o <= upper_bound and o >= lower_bound:
			test_energies_limited.append(o)
			predictions_limited.append(p)
			test_XS_limited.append(qx)

	plt.figure()
	plt.plot(unheated_energies_limited, unheated_XS_limited, label = '0 K JEFF-3.3')
	plt.grid()
	plt.plot(test_energies_limited, predictions_limited, label='Predictions', color='red')
	plt.xlabel('Energy / eV')
	plt.ylabel('$\sigma_{n,\gamma} / b$')
	plt.plot(test_energies_limited, test_XS_limited, '--', label=f'{test_temperatures[0]} K JEFF-3.3', color='lightgreen',
			 alpha=0.7)
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
	plt.savefig(f'{plotdir}/{timestring}-standard_multinodal_plot.png')
	plt.show()


	relativeError = []
	percentageError = []
	for p, xs in zip(predictions_limited, test_XS_limited):
		relativeError.append(abs((p-xs)/xs))
		percentageError.append((p/xs * 100) - 100)



	plt.figure()
	plt.plot(test_energies_limited, relativeError, label = 'Error')
	plt.xlabel('Energy / eV')
	plt.ylabel('Relative error')
	plt.xscale('log')
	plt.legend()
	plt.yscale('log')
	plt.grid()
	plt.show()

	plt.figure()
	plt.plot(test_energies_limited, percentageError, label='Error')
	plt.xlabel('Energy / eV')
	plt.ylabel('% Error')
	plt.grid()
	plt.savefig(f'{plotdir}/{timestring}-standard_multinodal_errors.png')
	plt.show()

	plt.figure()
	plt.hist(percentageError, bins=50)
	plt.ylabel('Frequency')
	plt.xlabel('% Error')
	plt.grid()
	plt.show()

	countoverthreshold = 0
	for XX in percentageError:
		if abs(XX) >= 0.1:
			countoverthreshold += 1

	percentageOverThreshold = (countoverthreshold / (len(percentageError))) * 100

	print(f'Max error: {np.max(abs(np.array(percentageError)))}')
	print(f'Mean error: {np.mean(abs(np.array(percentageError)))}')
	print(f'{percentageOverThreshold} % of points over limit of 0.1 % error')


bounds(minerg, maxerg)

plt.figure()
plt.plot(history.history['loss'])
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()




# import time
# time1 = time.time()
# itlength = 1000000
# for i in tqdm.tqdm(range(0,itlength)):
# 	tp = model.predict(X_test)
#
# time2 = time.time()
# elapsed = time2 - time1
# elapsedformatted = str(datetime.timedelta(seconds=elapsed))
# singleit = elapsed / itlength
# single_iter = str(datetime.timedelta(seconds=singleit))
# print(f'Elapsed: {elapsedformatted} - Time per inference: {single_iter}')