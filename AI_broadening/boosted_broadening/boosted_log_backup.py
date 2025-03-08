import os

os.environ["OMP_NUM_THREADS"] = "25"
os.environ["MKL_NUM_THREADS"] = "25"
os.environ["OPENBLAS_NUM_THREADS"] = "25"
os.environ["TF_NUM_INTEROP_THREADS"] = "25"
os.environ["TF_NUM_INTRAOP_THREADS"] = "25"


import random
import xgboost as xg
import pandas as pd
import matplotlib.pyplot as plt
import periodictable
import numpy as np
import scipy



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
	for o, p, qx in zip(rescaled_energies, rescaled_predictions, rescaled_test_xs):
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


df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
df = pd.read_csv('../AI_data/Fe56_200_to_1800_D1K_MT102.csv')
print('Data loaded')



# minerg = 700 # in eV
# maxerg = 1200 # in eV

minerg = 800 # in eV
maxerg = 1600 # in eV


df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

all_temperatures = np.arange(200, 1801, 1)


test_temperatures = [1500]
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

test_dataframe = df[df['T'].isin(test_temperatures)]
training_dataframe = df[df['T'].isin(training_temperatures)]
logged_T_train = np.log(training_dataframe['T'].values)
scaled_T_train = [(x - mean_alltemps) / std_alltemps for x in logged_T_train]
logged_ERG_train = np.log(training_dataframe['ERG'].values)
X_train = np.array([scipy.stats.zscore(logged_ERG_train), scaled_T_train])
X_train = np.transpose(X_train)
y_train_logged = np.array(np.log(training_dataframe['XS'].values))
y_train = scipy.stats.zscore(y_train_logged)




logged_T_test = np.log(test_dataframe['T'].values)
scaled_T_test = [(x - mean_alltemps) / std_alltemps for x in logged_T_test]

ERG_test = test_dataframe['ERG'].values
logged_ERG_test = np.log(ERG_test)
X_test = np.array([scipy.stats.zscore(logged_ERG_test), scaled_T_test])
X_test = np.transpose(X_test)
logged_y_test = np.log(np.array(test_dataframe['XS'].values))
y_test = scipy.stats.zscore(logged_y_test)


def log_loss_obj(y_pred, dtrain):
	y_true = dtrain.get_label()
	eps = 1e-15  # Prevent log(0)
	y_pred = np.clip(y_pred, eps, 1 - eps)
	grad = (y_pred - y_true) / (y_pred * (1 - y_pred))  # First derivative
	hess = (y_true - 2 * y_true * y_pred + y_pred**2) / (y_pred**2 * (1 - y_pred)**2)  # Second derivative
	return grad, hess



model = xg.XGBRegressor(n_estimators = 21800,
						max_depth = 16,
						learning_rate = 0.00039749142484026765,
						reg_lambda = 25.06,
						subsample = 0.9943057693219584,
						)


model.fit(X_train, y_train, verbose = True,
		  eval_set = [(X_train, y_train),
					  # (X_val, y_val),
					  (X_test, y_test)],
		  )


predictions = model.predict(X_test)
history = model.evals_result()


# test_energies = X_test.transpose()[0]
# test_energies = [e * 1e6 for e in test_energies]

unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg)) & (df0['ERG'] < (maxerg))]['ERG'].values
unheated_energies = [e for e in unheated_energies]
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg)) & (df0['ERG'] < (maxerg))]['XS'].values

scaled_energies = []
for pair in X_test:
	scaled_energies.append(pair[0])

rescaled_energies = np.array(scaled_energies) * np.std(logged_ERG_test) + np.mean(logged_ERG_test)
rescaled_energies = np.e ** rescaled_energies

rescaled_predictions = np.array(predictions) * np.std(logged_y_test) + np.mean(logged_y_test)
rescaled_predictions = np.e ** rescaled_predictions

rescaled_test_xs = np.array(y_test) * np.std(logged_y_test) + np.mean(logged_y_test)


# rescaled_test_energies = [np.e ** E for E in test_energies]
rescaled_test_xs = [np.e ** XS for XS in rescaled_test_xs]

# rescaled_predictions = [np.e ** p for p in predictions]

plt.figure()
plt.plot(np.array(unheated_energies), np.array(unheated_XS), label = '0 K JEFF-3.3')
plt.grid()
plt.plot(np.array(rescaled_energies), np.array(rescaled_predictions), label = 'Predictions', color = 'red')
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,\gamma} / b$')
plt.plot(np.array(rescaled_energies), np.array(rescaled_test_xs), '--', label = f'{test_temperatures[0]} K JEFF-3.3', color = 'lightgreen')
plt.legend()
plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
plt.yscale('log')
plt.xscale('log')
plt.savefig('testplot_highrange.png', dpi=300)
plt.show()





model.get_booster().feature_names = ['ERG', 'T']
plt.figure(figsize=(10, 12))
xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
plt.show()

plt.figure()
plt.plot(history['validation_0']['rmse'], label='Training loss')
plt.plot(history['validation_1']['rmse'], label='Validation loss')
plt.title('Loss plots')
plt.ylabel('RMSE / b')
plt.yscale('log')
plt.xlabel('N. Trees')
plt.grid()
plt.legend()
plt.show()



bounds(minerg, maxerg)