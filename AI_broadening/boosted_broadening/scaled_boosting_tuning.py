import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"
import random
import xgboost as xg
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hyperopt.early_stop
from hyperopt import  hp, fmin, tpe, STATUS_OK, Trials
import pickle
import scipy

df = pd.read_csv('../AI_data/Fe56_200_to_1800_D1K_MT102.csv')

ntreeguess = np.arange(100, 25000, 200)
depthguess = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

space = {'n_estimators': hp.choice('n_estimators', ntreeguess),
		 'subsample': hp.uniform('subsample', 0.01, 1.0),
		 'max_leaves': 0,
		 'max_depth': hp.choice('max_depth', depthguess),
		 'reg_lambda': hp.uniform('reg_lambda', 0, 100),
		 'gamma': hp.loguniform('gamma', 0, 40),
		 'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1))}


nuclide = [26, 56]
minerg = 600 # in eV
maxerg = 1500 # in eV


df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

all_temperatures = np.arange(200, 1801, 1)
all_temperatures = all_temperatures[all_temperatures != 1250]
test_temperatures = [1500]





def optimiser(space):

	print(space)

	validation_temperatures = []
	while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
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






	model = xg.XGBRegressor(**space, seed=42)

	model.fit(X_train, y_train, verbose=True,
			  eval_set=[  # (X_train, y_train),
				  # (X_val, y_val),
				  (X_test, y_test)],
			  early_stopping_rounds=20)

	predictions = model.predict(X_test)
	history = model.evals_result()

	scaled_energies = []
	for pair in X_test:
		scaled_energies.append(pair[0])

	rescaled_energies = np.array(scaled_energies) * np.std(logged_ERG_test) + np.mean(logged_ERG_test)
	rescaled_energies = np.e ** rescaled_energies

	rescaled_predictions = np.array(predictions) * np.std(logged_y_test) + np.mean(logged_y_test)
	rescaled_predictions = np.e ** rescaled_predictions

	rescaled_test_xs = np.array(y_test) * np.std(logged_y_test) + np.mean(logged_y_test)
	rescaled_test_xs = [np.e ** XS for XS in rescaled_test_xs]

	mae_loss = mean_absolute_error(rescaled_predictions, rescaled_test_xs)

	return {'loss': mae_loss, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=200,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(50))

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)

with open("best_xgboost_scaled_model_600to1500.pkl", "wb") as f:
	pickle.dump(best_model, f)