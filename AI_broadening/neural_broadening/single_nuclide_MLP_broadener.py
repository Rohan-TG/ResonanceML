import numpy as np
import tensorflow as tf
import keras
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from neural_broadening_functions import single_nuclide_make_train, single_nuclide_make_test, single_nuclide_make_val
import pandas as pd

df =pd.read_csv('Fe56_MT_102_Delta100K_0K_1800K.csv')

test_temperatures = [1800]
validation_temperatures = [1700,
						   1600,
						   1500,
						   1400,
						   # 1300,
						   # 1200,
						   # 1100,
						   ]
nuclide = [26,56]


min_energy = 100
max_energy = 2e6
X_train, y_train, unscaled_erg_train, unscaled_xs_train = single_nuclide_make_train(df=df,
											 val_temperatures=validation_temperatures,
											 test_temperatures=test_temperatures,
											 minERG=min_energy,
											 maxERG=max_energy,
											 use_tqdm=True)

X_test, y_test, unscaled_energy, unscaled_xs = single_nuclide_make_test(df=df,
										  test_temperatures=test_temperatures,
										  use_tqdm=True,
										  minERG=min_energy,
										  maxERG=max_energy)

X_val, y_val, unscaled_val_energies = single_nuclide_make_val(df=df,
										use_tqdm=True,
										minERG=min_energy,
										maxERG=max_energy,
										val_temperatures=validation_temperatures)

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 min_delta=0.01,
										 patience=5,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Dense(10, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=5,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)

predictions = model.predict(X_test)
predictions = predictions.ravel()



scaled_energies = []
for pair in X_test:
	scaled_energies.append(pair[0])


all_energies = df['ERG'].values
all_xs = df['XS'].values

max_all_energy = max(unscaled_energy) * 1e6
min_all_energy = min(unscaled_energy) * 1e6

max_all_xs = max(unscaled_xs)
min_all_xs = min(unscaled_xs)

rescaled_energies = np.array(scaled_energies)* (max_all_energy - min_all_energy) + min_all_energy

rescaled_test_xs = np.array(y_test) * (max_all_xs - min_all_xs) + min_all_xs


rescaled_predictions = np.array(predictions) * (max_all_xs - min_all_xs) + min_all_xs


unheated_energies = df[(df['T'] == 0) & (df['ERG'] > (min_energy/1e6)) & (df['ERG'] < (max_energy/1e6))]['ERG'].values
unheated_energies = [e * 1e6 for e in unheated_energies]
unheated_XS = df[(df['T'] == 0) & (df['ERG'] > (min_energy/1e6)) & (df['ERG'] < (max_energy/1e6))]['XS'].values


plt.figure()
plt.plot(rescaled_energies, rescaled_predictions, label = 'Predictions')
plt.plot(rescaled_energies, rescaled_test_xs, label = 'JEFF-3.3 1,800 K')
plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
plt.legend()
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,\gamma} / b$')
plt.xscale('log')
plt.yscale('log')
plt.show()
