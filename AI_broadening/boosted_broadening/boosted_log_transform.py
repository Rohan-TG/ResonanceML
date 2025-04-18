import os

os.environ["OMP_NUM_THREADS"] = "60"
os.environ["MKL_NUM_THREADS"] = "60"
os.environ["OPENBLAS_NUM_THREADS"] = "60"
os.environ["TF_NUM_INTEROP_THREADS"] = "60"
os.environ["TF_NUM_INTRAOP_THREADS"] = "60"


import random
import xgboost as xg
import pandas as pd
import matplotlib.pyplot as plt
import periodictable
import numpy as np
import tqdm
import datetime



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
	plt.savefig('preserror.png', dpi=300)
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
maxerg = 1400 # in eV


df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

all_temperatures = np.arange(200, 1801, 1)


test_temperatures = [1200]
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
X_train = np.array([np.log10(training_dataframe['ERG'].values), training_dataframe['T'].values])
X_train = np.transpose(X_train)
y_train = np.array(np.log10(training_dataframe['XS'].values))

X_test = np.array([np.log10(test_dataframe['ERG'].values), test_dataframe['T'].values])
X_test = np.transpose(X_test)
y_test = np.array(np.log10(test_dataframe['XS'].values))

def log_loss_obj(y_pred, dtrain):
	y_true = dtrain.get_label()
	eps = 1e-15  # Prevent log(0)
	y_pred = np.clip(y_pred, eps, 1 - eps)
	grad = (y_pred - y_true) / (y_pred * (1 - y_pred))  # First derivative
	hess = (y_true - 2 * y_true * y_pred + y_pred**2) / (y_pred**2 * (1 - y_pred)**2)  # Second derivative
	return grad, hess





model = xg.XGBRegressor(n_estimators = 11450,
						max_depth = 16,
						learning_rate = 0.0025919607000481934,
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

rescaled_test_energies = [10 ** E for E in test_energies]
rescaled_test_XS = [10 ** XS for XS in y_test]

rescaled_predictions = [10 ** p for p in predictions]

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
# plt.xscale('log')
plt.savefig('hightreeplot.png', dpi=300)
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

# import time
# time1 = time.time()
# itlength = 1000000
# testd = X_test[100]
# testd = testd.reshape(1,2)
# for i in tqdm.tqdm(range(0,itlength)):
# 	tp = model.predict(testd)
#
# time2 = time.time()
# elapsed = time2 - time1
# elapsedformatted = str(datetime.timedelta(seconds=elapsed))
# singleit = elapsed / itlength
# single_iter = str(datetime.timedelta(seconds=singleit))
# print(f'Elapsed: {elapsedformatted} - Time per inference: {single_iter}')