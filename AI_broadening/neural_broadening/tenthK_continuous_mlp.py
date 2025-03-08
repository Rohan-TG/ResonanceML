import pandas as pd
import keras
import datetime
import numpy as np
import random
import os
import scipy
import tqdm

minerg = 800
maxerg = 1500
test_temperatures = [1700.0]
nuclide = [26,56]

def get_datetime_string():
	return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

all_temperatures = np.arange(200, 3500, 0.1) # all temperatures in the data file

data_dir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT0.1k_single_temp'

df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values

validation_temperatures = []
while len(validation_temperatures) < int(len(all_temperatures) * 0.1):
	choice = random.choice(all_temperatures)
	if choice not in validation_temperatures and choice not in test_temperatures:
		validation_temperatures.append(choice)

training_temperatures = []
for T in all_temperatures:
	if T not in test_temperatures and T not in validation_temperatures:
		training_temperatures.append(T)

mean_alltemps = np.mean(all_temperatures)
std_alltemps = np.std(all_temperatures)


filenames = os.listdir(data_dir)

ERG_val = []
XS_val = []
T_val = []

ERG_train = []
XS_train = []
T_train = []

ERG_test = []
XS_test = []
T_test = []


for train_temperature in tqdm.tqdm(training_temperatures, total = len(training_temperatures)):
	filename = f'Fe_56_{train_temperature}.csv'
	df = pd.read_csv(f'{data_dir}/{filename}')

	ERG_train += df['ERG'].values
	XS_train += df['XS'].values
	T_train += df['T'].values

logged_T_train = np.log(T_train)
scaled_T_train = [(x - mean_alltemps) / std_alltemps for x in logged_T_train]
logged_ERG_train = np.log(ERG_train)
X_train = np.array([scipy.stats.zscore(logged_ERG_train), scaled_T_train])
X_train = np.transpose(X_train)
y_train_logged = np.array(np.log(XS_train))
y_train = scipy.stats.zscore(y_train_logged)


for test_temperature in tqdm.tqdm(test_temperatures, total=len(test_temperatures)):
	filename = f'Fe_56_{test_temperature}.csv'
	dftest = pd.read_csv(f'{data_dir}/{filename}')

	ERG_test += dftest['ERG'].values
	XS_test += dftest['XS'].values
	T_test += dftest['T'].values

logged_T_test = np.log(T_test)
scaled_T_test = [(x - mean_alltemps) / std_alltemps for x in logged_T_test]


logged_ERG_test = np.log(ERG_test)
X_test = np.array([scipy.stats.zscore(logged_ERG_test), scaled_T_test])
X_test = np.transpose(X_test)
logged_y_test = np.log(XS_test)
y_test = scipy.stats.zscore(logged_y_test)



callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=20,
										 mode='min',
										 start_from_epoch=5,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(500, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='adam')


history = model.fit(X_train,
					y_train,
					epochs=100,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_train, y_train),
					verbose=1)

predictions = model.predict(X_test)
predictions = predictions.ravel()

scaled_energies = []
for pair in X_test:
	scaled_energies.append(pair[0])

rescaled_energies = np.array(scaled_energies) * np.std(logged_ERG_test) + np.mean(logged_ERG_test)
rescaled_energies = np.e ** rescaled_energies

rescaled_predictions = np.array(predictions) * np.std(logged_y_test) + np.mean(logged_y_test)
rescaled_predictions = np.e ** rescaled_predictions

rescaled_test_xs = np.array(y_test) #* np.std(df['XS'].values) + np.mean(df['XS'].values)

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
	rescaled_test_XS = y
	for o, p, qx in zip(rescaled_energies, rescaled_predictions, rescaled_test_XS):
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
	# if scaley == 'log':
	# 	plt.yscale('log')
	# else:
	# 	plt.yscale('linear')
	# if scalex ==' log':
	# 	plt.xscale('log')
	# else:
	# 	plt.xscale('linear')
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


timestring = get_datetime_string()