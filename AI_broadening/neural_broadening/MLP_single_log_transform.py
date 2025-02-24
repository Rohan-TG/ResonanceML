import scipy.stats
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from neural_broadening_functions import log_single_nuclide_data_maker
import pandas as pd
import random
import periodictable

minerg = 800
maxerg = 1500






all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file


df = pd.read_csv('../AI_data/Fe56_200_to_1800_D1K_MT102.csv')

df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

# df = pd.read_csv('Fe56_200_to_1800_D1K.MT102.csv')

test_temperatures = [1400]
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

test_dataframe = df[df['T'].isin(test_temperatures)]
training_dataframe = df[df['T'].isin(training_temperatures)]
logged_T_train = np.log(training_dataframe['T'].values)
logged_ERG_train = np.log(training_dataframe['ERG'].values)
X_train = np.array([scipy.stats.zscore(logged_ERG_train), scipy.stats.zscore(logged_T_train)])
X_train = np.transpose(X_train)
y_train_logged = np.array(np.log(training_dataframe['XS'].values))
y_train = scipy.stats.zscore(y_train_logged)




logged_T_test = np.log(test_dataframe['T'].values)
ERG_test = test_dataframe['ERG'].values
logged_ERG_test = np.log(ERG_test)
X_test = np.array([scipy.stats.zscore(logged_ERG_test), scipy.stats.zscore(logged_ERG_test)])
X_test = np.transpose(X_test)
logged_y_test = np.log(np.array(test_dataframe['XS'].values))
y_test = scipy.stats.zscore(logged_y_test)


# X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, feature_means, feature_stds = log_single_nuclide_data_maker(df, val_temperatures=validation_temperatures,
# 																																								 test_temperatures=test_temperatures,
# 																																								 use_tqdm=True,
# 																																								 minERG=minerg,
# 																																								 maxERG=maxerg,)
#
#
#

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=10,
										 mode='min',
										 start_from_epoch=5,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
# model.add(keras.layers.LeakyReLU(alpha=0.05))
# model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu', bias_regularizer=keras.regularizers.L2(0.01)))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(200, activation='relu'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(20, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')

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




# logged_ERG_test = np.log(ERG_test)
# logged_y_test = np.log(y_test)

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
	rescaled_test_XS = test_dataframe['XS'].values
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



	# plt.figure()
	# plt.plot(test_energies_limited, relativeError, label = 'Error')
	# plt.xlabel('Energy / eV')
	# plt.ylabel('Relative error')
	# plt.xscale('log')
	# plt.legend()
	# plt.yscale('log')
	# plt.grid()
	# plt.show()

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
		if XX >= 0.1:
			countoverthreshold += 1

	percentageOverThreshold = (countoverthreshold / (len(percentageError))) * 100

	print(f'Max error: {np.max(abs(np.array(percentageError)))}')
	print(f'Mean error: {np.mean(abs(np.array(percentageError)))}')
	print(f'{percentageOverThreshold} % of points over limit of 0.1 % error')










plt.figure()
plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
plt.plot(ERG_test, test_dataframe['XS'].values, '--', label = 'JEFF-3.3 1,800 K')
plt.plot(rescaled_energies, rescaled_predictions, label = 'Predictions', color = 'red')
plt.legend()
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,\gamma} / b$')
plt.xscale('log')
plt.yscale('log')
plt.savefig('mlpplot.png', dpi = 300)
plt.show()

plt.figure()
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()