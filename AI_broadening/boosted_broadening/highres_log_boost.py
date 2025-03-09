import os
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["MKL_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
os.environ["TF_NUM_INTEROP_THREADS"] = "40"
os.environ["TF_NUM_INTRAOP_THREADS"] = "40"
import pandas as pd
import numpy as np
import periodictable
import datetime
import itertools
import random
import matplotlib.pyplot as plt
import tqdm
import xgboost as xg




minerg = 800 / 1e6
maxerg = 1500 / 1e6
test_temperatures = [1300.0]
nuclide = [26,56]

def get_datetime_string():
	return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

maxtemp = 1800
mintemp = 1600
numbers = np.linspace(mintemp, maxtemp, int((maxtemp - mintemp) / 0.1) + 1, dtype=np.float32) # all temperatures in the data file
all_temperatures = [round(NUM, 1) for NUM in numbers]
# all_temperatures = all_temperatures[all_temperatures != 254.7]

data_dir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT0.1k_single_temp'

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


exclusions = [254.7, 254.8, 254.9, 255.0]

for train_temperature in tqdm.tqdm(training_temperatures, total = len(training_temperatures)):
	if round(float(train_temperature), 1) not in exclusions:
		roundedtt = str(round(train_temperature, 1))
		filename = f'Fe_56_{roundedtt}K.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]
		ERG_train.append(df['ERG'].values)
		XS_train.append(df['XS'].values)
		T_train.append(df['T'].values)

ERG_train = list(itertools.chain(*ERG_train))
T_train = list(itertools.chain(*T_train))
XS_train = list(itertools.chain(*XS_train))

logged_T_train = np.log(T_train)
logged_ERG_train = np.log(ERG_train)
X_train = np.array([logged_ERG_train, logged_T_train])
X_train = np.transpose(X_train)
y_train = np.array(np.log(XS_train))



for test_temperature in tqdm.tqdm(test_temperatures, total=len(test_temperatures)):
	roundedtestt = str(round(test_temperature,1))
	filename = f'Fe_56_{roundedtestt}K.csv'
	dftest = pd.read_csv(f'{data_dir}/{filename}')
	dftest = dftest[(df['ERG'] < maxerg) & (dftest['ERG'] > minerg)]

	ERG_test.append(dftest['ERG'].values)
	XS_test.append(dftest['XS'].values)
	T_test.append(dftest['T'].values)

ERG_test = list(itertools.chain(*ERG_test))
T_test = list(itertools.chain(*T_test))
XS_test = list(itertools.chain(*XS_test))

logged_T_test = np.log(T_test)


logged_ERG_test = np.log(ERG_test)
X_test = np.array([logged_ERG_test, logged_T_test])
X_test = np.transpose(X_test)
logged_y_test = np.log(XS_test)
y_test = XS_test



model = xg.XGBRegressor(n_estimators = 10,
						max_depth = 11,
						learning_rate = 0.25919607000481934,
						reg_lambda = 2.415057075497998,
						subsample = 0.13021504261911765,
						)


model.fit(X_train, y_train, verbose = True,
		  eval_set = [(X_train, y_train),
					  # (X_val, y_val),
					  (X_test, y_test)],
		  )


predictions = model.predict(X_test)
history = model.evals_result()


test_energies = X_test.transpose()[0]
# test_energies = [e * 1e6 for e in test_energies]

unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg)) & (df0['ERG'] < (maxerg))]['ERG'].values
unheated_energies = [e for e in unheated_energies]
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg)) & (df0['ERG'] < (maxerg))]['XS'].values

rescaled_test_energies = [np.e ** E for E in test_energies]
rescaled_test_XS = [np.e ** XS for XS in y_test]

rescaled_predictions = [np.e ** p for p in predictions]

plt.figure()
plt.plot(np.array(unheated_energies), np.array(unheated_XS), label = '0 K JEFF-3.3')
plt.grid()
plt.plot(np.array(rescaled_test_energies), np.array(rescaled_predictions), label = 'Predictions', color = 'red')
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,\gamma} / b$')
plt.plot(np.array(rescaled_test_energies), np.array(rescaled_test_XS), '--', label = f'{test_temperatures[0]} K JEFF-3.3', color = 'lightgreen')
plt.legend()
plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
plt.yscale('log')
plt.xscale('log')
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
	for o, p, qx in zip(rescaled_test_energies, rescaled_predictions, rescaled_test_XS):
		if o <= upper_bound and o >= lower_bound:
			test_energies_limited.append(o)
			predictions_limited.append(p)
			test_XS_limited.append(qx)

	print(predictions_limited)
	print(test_XS_limited)

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
		if abs(XX) >= 0.1:
			countoverthreshold += 1

	percentageOverThreshold = (countoverthreshold / (len(percentageError))) * 100

	print(f'Max error: {np.max(abs(np.array(percentageError)))}')
	print(f'Mean error: {np.mean(abs(np.array(percentageError)))}')
	print(f'{percentageOverThreshold} % of points over limit of 0.1 % error')
	print(5)

print(5)


bounds(minerg, maxerg)
