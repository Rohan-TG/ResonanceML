import xgboost as xg
import pandas as pd
import matplotlib.pyplot as plt
import periodictable
from funcs import single_nuclide_make_train, single_nuclide_make_test


df = pd.read_csv('Fe56_MT_102_Delta100K_0K_1800K.csv')
print('Data loaded')

minerg = 500 # in eV
maxerg = 1 * 2.9e4 # in eV

test_temperatures = [500]
validation_temperatures = [1700,
						   1600,
						   1500,
						   1400,
						   # 1300,
						   # 1200,
						   # 1100,
						   ]
nuclide = [26,56]

X_train, y_train = single_nuclide_make_train(df=df,
											 val_temperatures=validation_temperatures,
											 test_temperatures=test_temperatures,
											 minERG=minerg,
											 maxERG=maxerg,
											 use_tqdm=True)

X_test, y_test = single_nuclide_make_test(df=df,
										  use_tqdm=True,
										  minERG=minerg,
										  maxERG=maxerg,
										  test_temperatures=test_temperatures)

X_val, y_val = single_nuclide_make_test(df=df,
										use_tqdm=True,
										minERG=minerg,
										maxERG=maxerg,
										test_temperatures=validation_temperatures)

progress = dict()

model = xg.XGBRegressor(n_estimators = 2000,
						max_depth = 9,
						learning_rate = 0.05,
						reg_lambda = 2
						)


model.fit(X_train, y_train, verbose = True,
		  eval_set = [(X_train, y_train), (X_val, y_val), (X_test, y_test)],)


predictions = model.predict(X_test)
history = model.evals_result()


test_energies = X_test.transpose()[0]
# test_energies = [e * 1e6 for e in test_energies]

unheated_energies = df[(df['T'] == 0) & (df['ERG'] > (minerg/1e6)) & (df['ERG'] < (maxerg/1e6))]['ERG'].values
unheated_energies = [e * 1e6 for e in unheated_energies]
unheated_XS = df[(df['T'] == 0) & (df['ERG'] > (minerg/1e6)) & (df['ERG'] < (maxerg/1e6))]['XS'].values


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

model.get_booster().feature_names = ['ERG', 'T']
plt.figure(figsize=(10, 12))
xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
plt.show()

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