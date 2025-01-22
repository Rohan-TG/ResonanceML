import xgboost as xg
import pandas as pd
import matplotlib.pyplot as plt
import periodictable
from funcs import single_nuclide_data_maker


df = pd.read_csv('Fe56_MT_102_Delta50K_0K_1800K.csv')
print('Data loaded')

minerg = 0 # in eV
maxerg = 1 * 2.9e5 # in eV





test_temperatures = [1500]
validation_temperatures = [1500,
						   # 1600,
						   # 1500,
						   # 1400,
						   1300,
						   # 1200,
						   1100,
						   ]
nuclide = [26,56]

X_train, y_train, X_val, y_val, X_test, y_test = single_nuclide_data_maker(df=df,
											 val_temperatures=validation_temperatures,
											 test_temperatures=test_temperatures,
											 minERG=minerg,
											 maxERG=maxerg,
											 use_tqdm=True)


progress = dict()

model = xg.XGBRegressor(n_estimators = 1300,
						max_depth = 11,
						learning_rate = 0.214,
						reg_lambda = 93,
						subsample = 0.55
						)


model.fit(X_train, y_train, verbose = True,
		  eval_set = [(X_train, y_train), (X_val, y_val), (X_test, y_test)],)


predictions = model.predict(X_test)
history = model.evals_result()


test_energies = X_test.transpose()[0]
# test_energies = [e * 1e6 for e in test_energies]

unheated_energies = df[(df['T'] == 0) & (df['ERG'] > (minerg)) & (df['ERG'] < (maxerg))]['ERG'].values
unheated_energies = [e for e in unheated_energies]
unheated_XS = df[(df['T'] == 0) & (df['ERG'] > (minerg)) & (df['ERG'] < (maxerg))]['XS'].values


plt.figure()
plt.plot(unheated_energies, unheated_XS, label = '0 K JEFF-3.3')
plt.grid()
plt.plot(test_energies, predictions, label = 'Predictions', color = 'red')
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,\gamma} / b$')
plt.plot(test_energies, y_test, '--', label = f'{test_temperatures[0]} K JEFF-3.3', color = 'lightgreen')
plt.legend()
plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
plt.yscale('log')
plt.xscale('log')
plt.show()


def bounds(lower_bound, upper_bound, ):
	unheated_energies_limited = []
	unheated_XS_limited = []
	for x, h in zip(unheated_energies, unheated_XS):
		if x <= upper_bound and x >= lower_bound:
			unheated_energies_limited.append(x)
			unheated_XS_limited.append(h)

	test_energies_limited = []
	predictions_limited = []
	test_XS_limited = []
	for o, p, qx in zip(test_energies, predictions, y_test):
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
	plt.plot(test_energies_limited, test_XS_limited, '--', label=f'{test_temperatures[0]} K JEFF-3.3', color='red')
	plt.legend()
	plt.title(f'{periodictable.elements[nuclide[0]]}-{nuclide[1]} $\sigma_{{n,\gamma}}$ at {test_temperatures[0]} K')
	plt.yscale('log')
	plt.xscale('log')
	plt.show()


	percentageError = []
	for p, xs in zip(predictions_limited, test_XS_limited):
		percentageError.append((p/xs * 100) - 100)

	plt.figure()
	plt.plot(test_energies_limited, percentageError, label = 'Error')
	plt.xlabel('Energy / eV')
	plt.ylabel('% Error')
	plt.xscale('log')
	plt.grid()
	plt.show()



# model.get_booster().feature_names = ['ERG', 'T']
# plt.figure(figsize=(10, 12))
# xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
# plt.show()

plt.figure()
plt.plot(history['validation_0']['rmse'], label='Training loss')
plt.plot(history['validation_1']['rmse'], label='Validation loss')
plt.title('Loss plots')
plt.ylabel('RMSE / b')
plt.xlabel('N. Trees')
plt.grid()
plt.legend()
plt.show()


def error_plotter(libraryXS, predictedXS, energyGrid):
	percentageError = []
	for p, xs in zip(predictedXS, libraryXS):
		percentageError.append((p/xs * 100) - 100)

	plt.figure()
	plt.plot(energyGrid, percentageError, label = 'Error')
	plt.xlabel('Energy / eV')
	plt.ylabel('% Error')
	plt.xscale('log')
	plt.grid()
	plt.show()

error_plotter(y_test, predictions, test_energies)