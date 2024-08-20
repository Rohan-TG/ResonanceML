import numpy as np
import matplotlib.pyplot as plt
import keras
import random

x = np.linspace(0, 50, 500)
y = np.cos(x) + 0.5*np.sin(2*x) + 0.7 * np.sin(0.4*x)
WINDOW_SIZE = 50

def datamaker(xvals, yvals, window_size = WINDOW_SIZE):
	X = []
	Y = []

	for i in range(len(xvals) - window_size):
		row = [[a] for a in yvals[i:i+window_size]]

		X.append(row)
		label = yvals[i+window_size]
		Y.append(label)

	return (np.array(X), np.array(Y))


bigx, bigy = datamaker(xvals=x, yvals=y)


X_train, y_train = bigx[:50], bigy[:50]
X_val, y_val = bigx[50:75], bigy[50:75]
X_test, y_test = bigx[75:], bigy[75:]

model = keras.Sequential()
model.add(keras.layers.InputLayer((WINDOW_SIZE,1)))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(32, 'relu'))
model.add(keras.layers.Dense(1, 'linear'))
model.compile(loss=keras.losses.MeanSquaredError(),
			  optimizer=keras.optimizers.Adam(learning_rate=0.01),
			  metrics=[keras.metrics.RootMeanSquaredError()])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)


testpred = model.predict(X_test).flatten()
trainpred = model.predict(X_train).flatten()

plt.figure()
plt.plot(x[len(x) - len(y_test):], y[len(y) - len(y_test):], label='True')
plt.plot(x[len(x) - len(y_test):], testpred, label = 'Predictions')
plt.legend()
plt.title('Test predictionss')
plt.show()

plt.figure()
plt.plot(x[WINDOW_SIZE:len(y_train)+WINDOW_SIZE], y_train, label='True')
plt.plot(x[WINDOW_SIZE:len(X_train) + WINDOW_SIZE], trainpred, label = 'Predictions')
plt.title('Training predictions')
plt.legend()
plt.show()