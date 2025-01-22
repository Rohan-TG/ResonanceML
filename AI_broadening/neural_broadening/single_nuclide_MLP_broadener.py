import tensorflow as tf
import keras
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import pandas as pd

min_energy = 200
max_energy = 10000

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
		if row['ERG'] > maxERG or row['ERG'] < minERG:
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

	normalised_T_train = []

	alltemps = df['T'].values
	maxtemp = max(alltemps)
	for x in T_train:
		normalised_T_train.append(x / maxtemp)

	X_train = np.array([ERG_train, normalised_T_train])
	y_train = np.array(XS_train)
	y_train = scipy.stats.zscore(y_train)

	feature_means = []
	feature_stds = []
	for j_idx, feature_list in enumerate(X_train[:-1]):
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

	########## Validation params

	ERG_val_mean = np.mean(ERG_val)
	ERG_val_std = np.std(ERG_val)

	T_val_mean = np.mean(T_val)
	T_val_std = np.std(T_val)

	XS_val_mean = np.mean(XS_val)
	XS_val_std = np.std(XS_val)

	########## Test params

	ERG_test_mean = np.mean(ERG_test)
	ERG_test_std = np.std(ERG_test)

	# T_test_mean = np.mean(T_test)
	# T_test_std = np.std(T_test)




	scaled_ERG_val = []
	scaled_T_val = []
	scaled_XS_val = []

	for v in ERG_val:
		scaled_ERG_val.append((v - ERG_val_mean) / ERG_val_std)

	for v in T_val:
		scaled_T_val.append(v / maxtemp)

	for v in XS_val:
		scaled_XS_val.append((v - XS_val_mean) / XS_val_std)

	X_val = np.array([scaled_ERG_val, scaled_T_val])
	X_val = np.transpose(X_val)
	y_val = np.array(scaled_XS_val)







	scaled_ERG_test = []
	scaled_T_test = []

	for v in ERG_test:
		scaled_ERG_test.append((v - ERG_test_mean) / ERG_test_std)

	for v in T_test:
		scaled_T_test.append(v / maxtemp)


	X_test = np.array([scaled_ERG_test, scaled_T_test])
	X_test = np.transpose(X_test)
	y_test = np.array(XS_test)

	return X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, feature_means, feature_stds



# def single_nuclide_data_maker(df, val_temperatures=[], test_temperatures=[], use_tqdm=False, minERG=0.0, maxERG=30e6):
# 	XS_train = []
# 	ERG_train = []
# 	T_train = []
#
# 	XS_val = []
# 	ERG_val = []
# 	T_val = []
#
# 	XS_test = []
# 	ERG_test = []
# 	T_test = []
#
# 	if use_tqdm:
# 		iterator = tqdm.tqdm(df.iterrows(), total=len(df))
# 	else:
# 		iterator = df.iterrows()
#
# 	for i, row in iterator:
# 		if row['ERG'] > maxERG or row['ERG'] < minERG:
# 			continue
# 		if row['T'] in val_temperatures:
# 			XS_val.append(row['XS'])
# 			ERG_val.append(row['ERG'])
# 			T_val.append(row['T'])
# 		if row['T'] in test_temperatures:
# 			XS_test.append(row['XS'])
# 			ERG_test.append(row['ERG'])
# 			T_test.append(row['T'])
# 		if row['T'] not in val_temperatures and row['T'] not in test_temperatures:
# 			XS_train.append(row['XS'])
# 			ERG_train.append(row['ERG'])
# 			T_train.append(row['T'])
#
# 	X_train = np.array([ERG_train, T_train])
# 	y_train = np.array(XS_train)
# 	y_train = scipy.stats.zscore(y_train)
#
# 	feature_means = []
# 	feature_stds = []
# 	for j_idx, feature_list in enumerate(X_train):
# 		X_train[j_idx] = scipy.stats.zscore(feature_list)
# 		feature_means.append(np.mean(feature_list))
# 		feature_stds.append(np.std(feature_list))
#
# 	X_train = np.transpose(X_train)
#
# 	ERG_mean = np.mean(df['ERG'].values)
# 	ERG_std = np.std(df['ERG'].values)
#
# 	T_mean = np.mean(df['T'].values)
# 	T_std = np.std(df['T'].values)
#
# 	XS_mean = np.mean(df['XS'].values)
# 	XS_std = np.std(df['XS'].values)
#
#
#
#
#
# 	scaled_ERG_val = []
# 	scaled_T_val = []
# 	scaled_XS_val = []
#
# 	for v in ERG_val:
# 		scaled_ERG_val.append((v - ERG_mean) / ERG_std)
#
# 	for v in T_val:
# 		scaled_T_val.append((v - T_mean) / T_std)
#
# 	for v in XS_val:
# 		scaled_XS_val.append((v - XS_mean) / XS_std)
#
# 	X_val = np.array([scaled_ERG_val, scaled_T_val])
# 	X_val = np.transpose(X_val)
# 	y_val = np.array(scaled_XS_val)
#
#
#
#
#
#
#
# 	scaled_ERG_test = []
# 	scaled_T_test = []
# 	scaled_XS_test = []
#
# 	for v in ERG_test:
# 		scaled_ERG_test.append((v - ERG_mean) / ERG_std)
#
# 	for v in T_test:
# 		scaled_T_test.append((v - T_mean) / T_std)
#
# 	for v in XS_test:
# 		scaled_XS_test.append((v - XS_mean) / XS_std)
#
# 	X_test = np.array([scaled_ERG_test, scaled_T_test])
# 	X_test = np.transpose(X_test)
# 	y_test = np.array(scaled_XS_test)
#
# 	return X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, XS_test, feature_means, feature_stds






# df =pd.read_csv('Fe56_MT_102_Delta50K_0K_1800K.csv')

df = pd.read_csv('Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
test_temperatures = [1400]
validation_temperatures = [#1700,
						   # 1600,
						   # 1500,
						   # 1400,
						   # 1300,
						   # 1200,
						   1100, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2140, 2160, 2180, 2200,
       2220, 2240, 2260, 2280, 2300, 2320, 2340, 2360, 2380, 2400, 2420,
       2440, 2460, 2480, 2500, 2520, 2540, 2560, 2580, 2600, 2620, 2640,
       2660, 2680, 2700, 2720, 2740, 2760, 2780, 2800, 2820, 2840, 2860,
       2880, 2900, 2920, 2940, 2960, 2980, 3000, 3020, 3040, 3060, 3080,
       3100, 3120, 3140, 3160, 3180, 3200, 3220, 3240, 3260, 3280, 3300,
       3320, 3340, 3360, 3380, 3400, 3420, 3440, 3460, 3480, 3500, 3520,
       3540, 3560, 3580, 3600, 3620, 3640, 3660, 3680, 3700, 3720, 3740,
       3760, 3780, 3800, 3820, 3840, 3860, 3880, 3900, 3920, 3940, 3960,
       3980, 4000
						   ]
nuclide = [26,56]




unheated_energies = df[(df['T'] == 0) & (df['ERG'] > min_energy) & (df['ERG'] < max_energy)]['ERG'].values
unheated_XS = df[(df['T'] == 0) & (df['ERG'] > min_energy) & (df['ERG'] < max_energy)]['XS'].values

X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, feature_means, feature_stds = single_nuclide_data_maker(df, val_temperatures=validation_temperatures,
																																								 test_temperatures=test_temperatures,
																																								 use_tqdm=True,
																																								 minERG=min_energy,
																																								 maxERG=max_energy,)




callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 min_delta=0.00005,
										 patience=100,
										 mode='min',
										 start_from_epoch=10,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(500, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu', bias_regularizer=keras.regularizers.L2(0.01)))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(600,activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=200,
					batch_size=64,
					callbacks=callback,
					# validation_data=(X_val, y_val),
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


rescaled_energies = np.array(scaled_energies) * np.std(ERG_test) + np.mean(ERG_test)

rescaled_predictions = np.array(predictions) * np.std(y_test) + np.mean(y_test)

rescaled_test_xs = np.array(y_test) #* np.std(df['XS'].values) + np.mean(df['XS'].values)





plt.figure()
plt.plot(rescaled_energies, rescaled_predictions, label = 'Predictions', color = 'red')
plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
plt.plot(ERG_test, y_test, '--', label = 'JEFF-3.3 1,800 K')
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