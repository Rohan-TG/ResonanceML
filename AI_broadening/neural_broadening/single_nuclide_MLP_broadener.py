import tensorflow as tf
import keras
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from neural_broadening_functions import single_nuclide_data_maker
import pandas as pd

min_energy = 800
max_energy = 1500


all_temperatures = np.arange(0, 1801, 20) # all temperatures in the data file


df = pd.read_csv('Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')

# df = pd.read_csv('Fe56_200_to_1800_D1K.MT102.csv')

test_temperatures = [1400]
validation_temperatures = []
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
model.add(keras.layers.Dense(100, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
# model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.LeakyReLU(alpha=0.05))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu', bias_regularizer=keras.regularizers.L2(0.01)))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(200, activation='relu'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=500,
					batch_size=64,
					callbacks=callback,
					validation_data=(X_train, y_train),
					verbose=1)

predictions = model.predict(X_test)
predictions = predictions.ravel()



scaled_energies = []
for pair in X_test:
	scaled_energies.append(pair[0])






rescaled_energies = np.array(scaled_energies) * np.std(ERG_test) + np.mean(ERG_test)

rescaled_predictions = np.array(predictions) * np.std(y_test) + np.mean(y_test)

rescaled_test_xs = np.array(y_test) #* np.std(df['XS'].values) + np.mean(df['XS'].values)





plt.figure()
plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
plt.plot(ERG_test, y_test, '--', label = 'JEFF-3.3 1,800 K')
plt.plot(rescaled_energies, rescaled_predictions, label = 'Predictions', color = 'red')
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