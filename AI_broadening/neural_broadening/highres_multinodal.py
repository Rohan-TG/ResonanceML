import os

os.environ["OMP_NUM_THREADS"] = "60"
os.environ["MKL_NUM_THREADS"] = "60"
os.environ["OPENBLAS_NUM_THREADS"] = "60"
os.environ["TF_NUM_INTEROP_THREADS"] = "60"
os.environ["TF_NUM_INTRAOP_THREADS"] = "60"

import pandas as pd
import keras
import datetime
import numpy as np
import random
import periodictable
import scipy
import matplotlib.pyplot as plt
import tqdm

def get_datetime_string():
	return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

nuclide = [26, 56]

maxtemp = 400
mintemp = 300

test_temperatures = [305.0, 320.0, 330.0, 345.0, 360.0, 375.0, 390.0, 398.0]

minerg = 10000 / 1e6
maxerg = 1000000 / 1e6


numbers = np.linspace(mintemp, maxtemp, int((maxtemp - mintemp) / 0.1) + 1, dtype=np.float32) # all temperatures in the data file
all_temperatures = [round(NUM, 1) for NUM in numbers]



data_dir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/interpolated_high_res_0.1K'

df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg * 1e6)) & (df0['ERG'] < (maxerg * 1e6))]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg * 1e6)) & (df0['ERG'] < (maxerg * 1e6))]['XS'].values

validation_temperatures = []
while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
	choice = random.choice(all_temperatures)
	if choice not in validation_temperatures and choice not in test_temperatures:
		validation_temperatures.append(choice)

training_temperatures = []
for T in all_temperatures:
	if T not in test_temperatures and T not in validation_temperatures:
		training_temperatures.append(T)

logged_temperatures = np.log10(np.array(all_temperatures).astype(np.float128))
mean_alltemps = np.mean(logged_temperatures.astype(np.float128), dtype=np.float128)
std_alltemps = np.std(logged_temperatures.astype(np.float128), dtype= np.float128)


filenames = os.listdir(data_dir)
# exclusions = [254.7, 254.8, 254.9, 255.0]
exclusions = []

train_input_matrix = [] # contains the energy grid and temperatures
train_labels_matrix = [] # contains the cross sections

for train_temperature in tqdm.tqdm(training_temperatures, total = len(training_temperatures)):
	if round(float(train_temperature), 1) not in exclusions:
		roundedtt = str(round(train_temperature, 1))
		filename = f'Fe_56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		logged_T_values = np.log10(df['T'].values)
		scaled_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_T_values]

		input_submatrix = np.array(scaled_T_values)  # can add or remove ERG here to make energy an input parameter
		labelsubmatrix = np.array(np.log10(df['XS'].values))

		train_input_matrix.append(input_submatrix)
		train_labels_matrix.append(labelsubmatrix)

X_train = np.array(train_input_matrix)
y_train = np.array(train_labels_matrix)

flipped_y_train = y_train.transpose()

scaled_y_train = []

means_test = []
stds_test = []

for energyPointvalues in flipped_y_train:
	scaled_values = scipy.stats.zscore(energyPointvalues)
	scaled_y_train.append(scaled_values)

	means_test.append(np.mean(energyPointvalues))
	stds_test.append(np.std(energyPointvalues))


y_train = np.array(scaled_y_train)
y_train = y_train.transpose()


val_input_matrix = [] # contains the energy grid and temperatures
val_labels_matrix = [] # contains the cross sections

for val_temperature in tqdm.tqdm(validation_temperatures, total=len(validation_temperatures)):
	if round(float(val_temperature), 1) not in exclusions:
		roundedtt = str(round(val_temperature, 1))
		filename = f'Fe_56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		logged_val_T_values = np.log10(df['T'].values)
		scaled_val_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_val_T_values]

		input_submatrix_val = np.array(scaled_val_T_values)  # can add or remove ERG here to make energy an input parameter
		labelsubmatrix_val = np.array(np.log10(df['XS'].values))

		val_input_matrix.append(input_submatrix_val)
		val_labels_matrix.append(labelsubmatrix_val)

X_val = np.array(val_input_matrix)
y_val = np.array(val_labels_matrix)

flipped_y_val = y_val.transpose()

scaled_y_val = []
for energyPointvalues in flipped_y_val:
	scaled_values = scipy.stats.zscore(energyPointvalues)
	scaled_y_val.append(scaled_values)

y_val = np.array(scaled_y_val)
y_val = y_val.transpose()



test_input_matrix = []
test_labels_matrix = []

for test_temperature in tqdm.tqdm(test_temperatures, total=len(test_temperatures)):
	if round(float(test_temperature), 1) not in exclusions:
		roundedtt = str(round(test_temperature, 1))
		filename = f'Fe_56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		logged_test_T_values = np.log10(df['T'].values)
		scaled_test_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_test_T_values]

		input_submatrix_test = np.array(scaled_test_T_values)  # can add or remove ERG here to make energy an input parameter
		labelsubmatrix_test = np.array(np.log10(df['XS'].values))

		test_input_matrix.append(input_submatrix_test)
		test_labels_matrix.append(labelsubmatrix_test)


X_test = np.array(test_input_matrix)
y_test = np.array(test_labels_matrix)


scaled_y_test = []
for energyPointvalue, mean, std in zip(y_test, means_test, stds_test):
	scaled_value = (energyPointvalue - mean) / std
	scaled_y_test.append(scaled_value)

y_test = np.array(scaled_y_test)


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=10,
										 mode='min',
										 start_from_epoch=5,
										 restore_best_weights=True)



model = keras.Sequential()
model.add(keras.layers.Dense(2400, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(2200))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(1900))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(1700))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(1300))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(1000))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(16))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(y_test.shape[1], activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')
#
#
# # new additions
#
history = model.fit(X_train,
					y_train,
					epochs=200,
					batch_size=16,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)


predictions = model.predict(X_test)
predictions = predictions.ravel()

teststring = f'Fe_56_{int(test_temperatures[0])}.0.csv'
dftest = pd.read_csv(f'{data_dir}/{teststring}')
dftest = dftest[(dftest['ERG'] < maxerg) & (dftest['ERG'] > minerg)]

testxs = dftest['XS'].values


energies = dftest['ERG'].values

rescaled_predictions = []
for P, mean, std in zip(predictions, means_test, stds_test):
	descaled_p = P * std + mean
	rescaled_predictions.append(descaled_p)

rescaled_predictions = [10 ** p for p in rescaled_predictions]


def bounds(lower_bound, upper_bound, scalex='log', scaley='log'):
	unheated_energies_limited = []
	unheated_XS_limited = []
	for x, h in zip(unheated_energies, unheated_XS):
		if x <= upper_bound * 1e6 and x >= lower_bound * 1e6:
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

	# plt.figure()
	# plt.plot(unheated_energies_limited, unheated_XS_limited, label = '0 K JEFF-3.3')
	# plt.grid()
	# plt.plot(test_energies_limited, predictions_limited, label='Predictions', color='red')
	# plt.xlabel('Energy / eV')
	# plt.ylabel('$\sigma_{n,\gamma} / b$')
	# plt.plot(test_energies_limited, test_XS_limited, '--', label=f'{test_temperatures[0]} K JEFF-3.3', color='lightgreen',
	# 		 alpha=0.7)
	# plt.legend()
	# plt.xscale('log')
	# plt.yscale('log')
	# plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
	# plt.show()


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
	# plt.savefig(f'{plotdir}/{timestring}-multinodal_rel_error.png', dpi=300)
	plt.show()

	plt.figure()
	plt.plot(test_energies_limited, percentageError, label='Error')
	plt.xlabel('Energy / eV')
	plt.ylabel('% Error')
	plt.grid()
	# plt.savefig(f'{plotdir}/{timestring}-pct_error.png', dpi=300)
	plt.show()

	plt.figure()
	plt.hist(percentageError, bins=50)
	plt.ylabel('Frequency')
	plt.xlabel('% Error')
	plt.grid()
	plt.savefig(f'{plotdir}/{timestring}-error_histogram.png', dpi=300)
	plt.show()

	countoverthreshold = 0
	for XX in percentageError:
		if abs(XX) >= 0.1:
			countoverthreshold += 1

	percentageOverThreshold = (countoverthreshold / (len(percentageError))) * 100

	print(f'Max error: {np.max(abs(np.array(percentageError)))}')
	print(f'Mean error: {np.mean(abs(np.array(percentageError)))}')
	print(f'{percentageOverThreshold} % of points over limit of 0.1 % error')


# bounds(minerg, maxerg)

# plt.figure()
# plt.plot(history.history['loss'])
# plt.grid()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

# unheated_energies_limited = []
# unheated_XS_limited = []
# for x, h in zip(unheated_energies, unheated_XS):
# 	if x <= maxerg and x >= minerg:
# 		unheated_energies_limited.append(x)
# 		unheated_XS_limited.append(h)

test_energies_limited = []
predictions_limited = []
test_XS_limited = []
for o, p, qx in zip(energies, rescaled_predictions, testxs):
	if o <= maxerg and o >= minerg:
		test_energies_limited.append(o)
		predictions_limited.append(p)
		test_XS_limited.append(qx)


# relativeError = []
# percentageError = []
# for p, xs in zip(predictions_limited, test_XS_limited):
# 	relativeError.append(abs((p-xs)/xs))
# 	percentageError.append((p/xs * 100) - 100)
#
# print('Percentage error:', percentageError)
plotdir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/neural_broadening/multinodalplots'

# plt.figure()
# plt.plot(test_energies_limited, percentageError, label='Error')
# plt.xlabel('Energy / eV')
# plt.ylabel('% Error')
# # plt.savefig(f'{plotdir}/{timestring}-highres_multinodal_errors.png')
# plt.grid()
# plt.show()
timestring = get_datetime_string()
plt.figure()
# plt.plot(unheated_energies_limited, unheated_XS_limited, label='0 K JEFF-3.3')
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
# plt.savefig(f'{plotdir}/{timestring}-multinodal_plot.png', dpi=300)
plt.show()

bounds(minerg, maxerg)
