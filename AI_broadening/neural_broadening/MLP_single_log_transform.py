import tensorflow as tf
import keras
import scipy
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import pandas as pd





minerg = 800
maxerg = 1500

all_temperatures = np.arange(200, 1801, 1) # all temperatures in the data file

df0 = pd.read_csv('Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > minerg) & (df0['ERG'] < maxerg)]['XS'].values

df = pd.read_csv('Fe56_200_to_1800_D1K_MT102.csv')

df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

test_temperatures = [1400]
validation_temperatures = []
while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
	choice = random.choice(all_temperatures)
	if choice not in validation_temperatures and choice not in test_temperatures:
		validation_temperatures.append(choice)

training_temperatures = []
for T in all_temperatures:
	if T not in test_temperatures and T not in validation_temperatures:
		training_temperatures.append(T)
nuclide = [26,56]


test_dataframe = df[df['T'].isin(test_temperatures)]
training_dataframe = df[df['T'].isin(training_temperatures)]
X_train = np.array([np.log(training_dataframe['ERG'].values), np.log(training_dataframe['T'].values)])
X_train = np.transpose(X_train)
y_train = np.array(np.log(training_dataframe['XS'].values))

X_test = np.array([np.log(test_dataframe['ERG'].values), np.log(test_dataframe['T'].values)])
X_test = np.transpose(X_test)
y_test = np.array(np.log(test_dataframe['XS'].values))


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 min_delta=0.00005,
										 patience=100,
										 mode='min',
										 start_from_epoch=10,
										 restore_best_weights=True)


model = keras.Sequential()
model.add(keras.layers.Dense(100, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.LeakyReLU(alpha=0.05))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu', bias_regularizer=keras.regularizers.L2(0.01)))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dropout(0.05))
model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')


history = model.fit(X_train,
					y_train,
					epochs=50,
					batch_size=64,
					callbacks=callback,
					validation_data=(X_train, y_train),
					verbose=1)

predictions = model.predict(X_test)
predictions = predictions.ravel()


scaled_energies = []
for pair in X_test:
	scaled_energies.append(pair[0])


test_energies = X_test.transpose()[0]

rescaled_energies = [np.e ** ERG for ERG in scaled_energies]
rescaled_predictions = [np.e ** P for P in predictions]
rescaled_test_xs = [np.e ** TXS for TXS in y_test]
rescaled_test_energies = [np.e ** TERG for TERG in test_energies]

plt.figure()
plt.plot(unheated_energies, unheated_XS, label = 'JEFF-3.3 0 K')
plt.plot(rescaled_test_energies, rescaled_test_xs, '--', label = f'JEFF-3.3 {test_temperatures[0]} K')
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