import os
os.environ["OMP_NUM_THREADS"] = "60"
os.environ["MKL_NUM_THREADS"] = "60"
os.environ["OPENBLAS_NUM_THREADS"] = "60"
os.environ["TF_NUM_INTEROP_THREADS"] = "60"
os.environ["TF_NUM_INTRAOP_THREADS"] = "60"

import pandas as pd
import numpy as np
import hyperopt.early_stop
from hyperopt import  hp, fmin, tpe, STATUS_OK, Trials
import pickle
import tqdm
import random
import xgboost as xg
from sklearn.metrics import mean_absolute_error
import datetime


ntreeguess = np.arange(1000, 40000, 500)
depthguess = [2,3,4,5,6,7,8]

space = {'n_estimators': hp.choice('n_estimators', ntreeguess),
		 'subsample': hp.uniform('subsample', 0.01, 1.0),
		 'max_leaves': 0,
		 'max_depth': hp.choice('max_depth', depthguess),
		 'reg_lambda': hp.uniform('reg_lambda', 0, 100),
		 'gamma': hp.loguniform('gamma', 0, 40),
		 'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1))}




def get_datetime_string():
	return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")




minerg = 1000 # in eV
maxerg = 50000 # in eV

data_directory = '/home/rnt26/NJOY/data/Fe56_JEFF/CSVs'
print('Data loaded')



all_temperatures = np.arange(1000.0, 1300.0, 0.1)


def optimiser(space):
	test_temperatures = []
	while len(test_temperatures) < int(len(all_temperatures) * 0.1):
		testChoice = random.choice(all_temperatures)
		if testChoice not in test_temperatures:
			test_temperatures.append(testChoice)

	validation_temperatures = []
	while len(validation_temperatures) < int(len(all_temperatures) * 0.1):
		choice = random.choice(all_temperatures)
		if choice not in validation_temperatures and choice not in test_temperatures:
			validation_temperatures.append(choice)

	training_temperatures = []
	for T in all_temperatures:
		if T not in test_temperatures and T not in validation_temperatures:
			training_temperatures.append(T)


	# logged_temperatures = np.log10(np.array(all_temperatures).astype(np.float128))
	# mean_alltemps = np.mean(logged_temperatures.astype(np.float128), dtype=np.float128)
	# std_alltemps = np.std(logged_temperatures.astype(np.float128), dtype=np.float128)

	# filenames = os.listdir(data_directory)

	exclusions = []

	T_train = []
	XS_train = []
	ERG_train = []

	for train_temperature in tqdm.tqdm(training_temperatures, total=len(training_temperatures)):
		if round(float(train_temperature), 1) not in exclusions:
			roundedtt = str(round(train_temperature, 1))
			filename = f'Fe56_{roundedtt}.csv'
			df = pd.read_csv(f'{data_directory}/{filename}')
			df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

			logged_T_values = np.log10(df['T'].values)

			T_train.append(logged_T_values)  # can add or remove ERG here to make energy an input parameter
			XS_train.append(df['XS'].values)
			ERG_train.append(df['ERG'].values)

	flat_T_Train = [item for sublist in T_train for item in sublist]
	flat_ERG_train = [item for sublist in ERG_train for item in sublist]
	flat_XS_train = [item for sublist in XS_train for item in sublist]

	logged_erg_train = np.log10(flat_ERG_train)

	logged_xs_train = np.log10(flat_XS_train)

	X_train = np.array([logged_erg_train, flat_T_Train])
	X_train = X_train.transpose()
	y_train = np.array(logged_xs_train)

	# Validation data
	T_val = []
	XS_val = []
	ERG_val = []

	for validation_temperature in tqdm.tqdm(validation_temperatures, total=len(validation_temperatures)):
		if round(float(validation_temperature), 1) not in exclusions:
			roundedtt = str(round(validation_temperature, 1))
			filename = f'Fe56_{roundedtt}.csv'
			df = pd.read_csv(f'{data_directory}/{filename}')
			df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

			logged_T_values = np.log10(df['T'].values)

			T_val.append(logged_T_values)  # can add or remove ERG here to make energy an input parameter
			XS_val.append(df['XS'].values)
			ERG_val.append(df['ERG'].values)

	flat_T_val = [item for sublist in T_val for item in sublist]
	flat_ERG_val = [item for sublist in ERG_val for item in sublist]
	flat_XS_val = [item for sublist in XS_val for item in sublist]

	logged_erg_val = np.log10(flat_ERG_val)

	logged_xs_val = np.log10(flat_XS_val)

	X_val = np.array([logged_erg_val, flat_T_val])
	X_val = X_val.transpose()
	y_val = np.array(logged_xs_val)

	# Test data
	T_test = []
	XS_test = []
	ERG_test = []

	for test_temperature in tqdm.tqdm(test_temperatures, total=len(test_temperatures)):
		if round(float(test_temperature), 1) not in exclusions:
			roundedtt = str(round(test_temperature, 1))
			filename = f'Fe56_{roundedtt}.csv'
			df = pd.read_csv(f'{data_directory}/{filename}')
			df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

			logged_T_values = np.log10(df['T'].values)

			T_test.append(logged_T_values)
			XS_test.append(df['XS'].values)
			ERG_test.append(df['ERG'].values)

	flat_T_test = [item for sublist in T_test for item in sublist]
	flat_ERG_test = [item for sublist in ERG_test for item in sublist]
	flat_XS_test = [item for sublist in XS_test for item in sublist]

	logged_erg_test = np.log10(flat_ERG_test)

	logged_xs_test = np.log10(flat_XS_test)

	X_test = np.array([logged_erg_test, flat_T_test])
	X_test = X_test.transpose()
	y_test = np.array(logged_xs_test)

	# model = xg.XGBRegressor(n_estimators = 51450,
	# 						max_depth = 6,
	# 						learning_rate = 0.0025919607000481934,
	# 						reg_lambda = 2.415057075497998,
	# 						subsample = 0.13021504261911765,
	# 						)

	model = xg.XGBRegressor(**space,
							seed = 42)

	model.fit(X_train, y_train, verbose=True,
			  eval_set=[# (X_train, y_train),
						# (X_val, y_val),
						(X_val, y_val)],
			  early_stopping_rounds = 50
			  )

	predictions = model.predict(X_test)
	history = model.evals_result()

	test_energies = X_test.transpose()[0]
	# test_energies = [e * 1e6 for e in test_energies]


	rescaled_test_energies = [10 ** E for E in test_energies]
	rescaled_test_XS = [10 ** XS for XS in y_test]

	rescaled_predictions = [10 ** p for p in predictions]

	mae_loss = mean_absolute_error(rescaled_predictions, rescaled_test_XS)

	def bounds(lower_bound, upper_bound, scalex='log', scaley='log'):

		test_energies_limited = []
		predictions_limited = []
		test_XS_limited = []
		for o, p, qx in zip(rescaled_test_energies, rescaled_predictions, rescaled_test_XS):
			if o <= upper_bound and o >= lower_bound:
				test_energies_limited.append(o)
				predictions_limited.append(p)
				test_XS_limited.append(qx)

		relativeError = []
		percentageError = []
		for p, xs in zip(predictions_limited, test_XS_limited):
			relativeError.append(abs((p - xs) / xs))
			percentageError.append((p / xs * 100) - 100)

		countoverthreshold = 0
		for XX in percentageError:
			if XX >= 0.1:
				countoverthreshold += 1

		percentageOverThreshold = (countoverthreshold / (len(percentageError))) * 100

		print(f'Max error: {np.max(abs(np.array(percentageError)))}')
		print(f'Mean error: {np.mean(abs(np.array(percentageError)))}')
		print(f'{percentageOverThreshold} % of points over limit of 0.1 % error')

	bounds(minerg, maxerg)

	return {'loss': mae_loss, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=200,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(30))

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)

time = get_datetime_string()

model_dir = "/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/boosted_broadening/models"

with open(f"{model_dir}/{time}_best_new_xgboost_model.pkl", "wb") as f:
	pickle.dump(best_model, f)