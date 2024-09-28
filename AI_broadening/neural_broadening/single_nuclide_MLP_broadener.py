import tensorflow as tf
import keras
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import pandas as pd


def single_nuclide_data_maker(df, val_temperatures=[], test_temperatures=[], use_tqdm=False, minERG=0, maxERG=30e6):
	XS_train = []
	ERG_train = []
	T_train = []

	XS_val = []
	ERG_val = []
	T_val = []

	XS_test = []
	ERG_test = []
	T_test = []

	if use_tqdm:
		iterator = tqdm.tqdm(df.iterrows(), total=len(df))
	else:
		iterator = df.iterrows()

	for i, row in iterator:
		if (row['ERG'] * 1e6) > maxERG or (row['ERG'] * 1e6) < minERG:
			continue
		if row['T'] in val_temperatures:
			XS_val.append(row['XS'])
			ERG_val.append(row['ERG'])
			T_val.append(row['T'])
		if row['T'] in test_temperatures:
			XS_test.append(row['XS'])
			ERG_test.append(row['ERG'])
			T_test.append(row['T'])
		if row['T'] not in val_temperatures and row['T'] not in test_temperatures:
			XS_train.append(row['XS'])
			ERG_train.append(row['ERG'])
			T_train.append(row['T'])

	X_train = np.array([ERG_train, T_train])
	y_train = np.array(XS_train)
	y_train = scipy.stats.zscore(y_train)

	feature_means = []
	feature_stds = []
	for j_idx, feature_list in enumerate(X_train):
		X_train[j_idx] = scipy.stats.zscore(feature_list)
		feature_means.append(np.mean(feature_list))
		feature_stds.append(np.std(feature_list))

	X_train = np.transpose(X_train)

	ERG_train_mean = np.mean(ERG_train)
	ERG_train_std = np.std(ERG_train)

	T_train_mean = np.mean(T_train)
	T_train_std = np.std(T_train)

	XS_train_mean = np.mean(XS_train)
	XS_train_std = np.std(XS_train)





	scaled_ERG_val = []
	scaled_T_val = []
	scaled_XS_val = []

	for v in ERG_val:
		scaled_ERG_val.append((v - ERG_train_mean) / ERG_train_std)

	for v in T_val:
		scaled_T_val.append((v - T_train_mean) / T_train_std)

	for v in XS_val:
		scaled_XS_val.append((v - XS_train_mean) / XS_train_std)

	X_val = np.array([scaled_ERG_val, scaled_T_val])
	X_val = np.transpose(X_val)
	y_val = np.array(scaled_XS_val)







	scaled_ERG_test = []
	scaled_T_test = []
	scaled_XS_test = []

	for v in ERG_test:
		scaled_ERG_test.append((v - ERG_train_mean) / ERG_train_std)

	for v in T_test:
		scaled_T_test.append((v - T_train_mean) / T_train_std)

	for v in XS_test:
		scaled_XS_test.append((v - XS_train_mean) / XS_train_std)

	X_test = np.array([scaled_ERG_test, scaled_T_test])
	X_test = np.transpose(X_test)
	y_test = np.array(scaled_XS_test)

	return X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, XS_test, feature_means, feature_stds


df =pd.read_csv('Fe56_MT_102_Delta100K_0K_1800K.csv')

test_temperatures = [1800]
validation_temperatures = [1700,
						   1600,
						   # 1500,
						   1400,
						   1300,
						   1200,
						   # 1100,
						   ]
nuclide = [26,56]


min_energy = 0.00001
max_energy = 1e4
# X_train, y_train, unscaled_erg_train, unscaled_xs_train = single_nuclide_make_train(df=df,
# 											 val_temperatures=validation_temperatures,
# 											 test_temperatures=test_temperatures,
# 											 minERG=min_energy,
# 											 maxERG=max_energy,
# 											 use_tqdm=True)
#
# X_test, y_test, unscaled_energy, unscaled_xs = single_nuclide_make_test(df=df,
# 										  test_temperatures=test_temperatures,
# 										  use_tqdm=True,
# 										  minERG=min_energy,
# 										  maxERG=max_energy)
#
# X_val, y_val, unscaled_val_energies = single_nuclide_make_val(df=df,
# 										use_tqdm=True,
# 										minERG=min_energy,
# 										maxERG=max_energy,
# 										val_temperatures=validation_temperatures)
unheated_energies = df[(df['T'] == 0) & (df['ERG'] > (min_energy/1e6)) & (df['ERG'] < (max_energy/1e6))]['ERG'].values
unheated_energies = [e * 1e6 for e in unheated_energies]
unheated_XS = df[(df['T'] == 0) & (df['ERG'] > (min_energy/1e6)) & (df['ERG'] < (max_energy/1e6))]['XS'].values

X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, XS_test, feature_means, feature_stds = single_nuclide_data_maker(df, val_temperatures=validation_temperatures,
																																								 test_temperatures=test_temperatures,
																																								 use_tqdm=True, minERG=min_energy, maxERG=max_energy,)

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 min_delta=0.0001,
										 patience=50,
										 mode='min',
										 start_from_epoch=20,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(500, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu', bias_regularizer=keras.regularizers.L2(0.01)))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(600,activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=500,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)

predictions = model.predict(X_test)
predictions = predictions.ravel()



scaled_energies = []
for pair in X_test:
	scaled_energies.append(pair[0])



# max_test_energy = max(unscaled_energy) * 1e6
# min_test_energy = min(unscaled_energy) * 1e6
#
# max_test_xs = max(unscaled_xs)
# min_test_xs = min(unscaled_xs)
#
# max_train_xs = max(unscaled_xs_train)
# min_train_xs = min(unscaled_xs_train)
#
# max_train_erg = max(unscaled_erg_train)
# min_train_erg = min(unscaled_erg_train)



# rescaled_energies = np.array(scaled_energies)* (max_test_energy - min_test_energy) + min_test_energy
#
# rescaled_test_xs = np.array(y_test) * (max_test_xs - min_test_xs) + min_test_xs
#
# rescaled_predictions = np.array(predictions) * (max_test_xs - min_test_xs) + min_test_xs


rescaled_energies = 1e6 * np.array(scaled_energies) * np.std(ERG_train) + np.mean(ERG_train) * 1e6

rescaled_predictions = np.array(predictions) * np.std(XS_train) + np.mean(XS_train)

rescaled_test_xs = np.array(y_test) * np.std(XS_train) + np.mean(XS_train)





plt.figure()
plt.plot(rescaled_energies, rescaled_predictions, label = 'Predictions', color = 'red')
plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
plt.plot(rescaled_energies, rescaled_test_xs, '--', label = 'JEFF-3.3 1,800 K')
plt.legend()
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,\gamma} / b$')
plt.xscale('log')
plt.yscale('log')
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