# Bayesian HP optimisation script
import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["TF_NUM_INTEROP_THREADS"] = "20"
os.environ["TF_NUM_INTRAOP_THREADS"] = "20"


import random
from funcs import single_nuclide_data_maker
import xgboost as xg
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hyperopt.early_stop
from hyperopt import  hp, fmin, tpe, STATUS_OK, Trials
import pickle

df = pd.read_csv('../AI_data/Fe56_200_to_1800_D1K_MT102.csv')




ntreeguess = np.arange(3000, 25000, 200)
depthguess = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

space = {'n_estimators': hp.choice('n_estimators', ntreeguess),
		 'subsample': hp.uniform('subsample', 0.01, 1.0),
		 'max_leaves': 0,
		 'max_depth': hp.choice('max_depth', depthguess),
		 'reg_lambda': hp.uniform('reg_lambda', 0, 100),
		 'gamma': hp.loguniform('gamma', 0, 40),
		 'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1))}


nuclide = [26, 56]
minerg = 500 # in eV
maxerg = 26000 # in eV

df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

all_temperatures = np.arange(200, 1801, 1)
all_temperatures = all_temperatures[all_temperatures != 1250]
test_temperatures = [1500]
def optimiser(space):

	validation_temperatures = []
	while len(validation_temperatures) < int(len(all_temperatures) * 0.2):
		choice = random.choice(all_temperatures)
		if choice not in validation_temperatures and choice not in test_temperatures:
			validation_temperatures.append(choice)

	training_temperatures = []
	for T in all_temperatures:
		if T not in test_temperatures and T not in validation_temperatures:
			training_temperatures.append(T)

	# X_train, y_train, X_val, y_val, X_test, y_test = single_nuclide_data_maker(df=df,
	# 											 val_temperatures=validation_temperatures,
	# 											 test_temperatures=test_temperatures,
	# 											 minERG=minerg,
	# 											 maxERG=maxerg,
	# 											 use_tqdm=True)

	test_dataframe = df[df['T'].isin(validation_temperatures)]
	training_dataframe = df[df['T'].isin(training_temperatures)]
	X_train = np.array([np.log(training_dataframe['ERG'].values), training_dataframe['T'].values])
	X_train = np.transpose(X_train)
	y_train = np.array(np.log(training_dataframe['XS'].values))

	X_test = np.array([np.log(test_dataframe['ERG'].values), test_dataframe['T'].values])
	X_test = np.transpose(X_test)
	y_test = np.array(np.log(test_dataframe['XS'].values))

	model = xg.XGBRegressor(**space, seed=42)

	model.fit(X_train, y_train, verbose=True,
			  eval_set=[#(X_train, y_train),
						# (X_val, y_val),
						(X_test, y_test)],
			  early_stopping_rounds= 20)

	predictions = model.predict(X_test)

	# test_energies = X_test.transpose()[0]
	# test_energies = [e * 1e6 for e in test_energies]

	# unheated_energies = df[(df['T'] == 0) & (df['ERG'] > (minerg)) & (df['ERG'] < (maxerg))]['ERG'].values
	# unheated_energies = [e for e in unheated_energies]
	# unheated_XS = df[(df['T'] == 0) & (df['ERG'] > (minerg)) & (df['ERG'] < (maxerg))]['XS'].values

	# rescaled_test_energies = [np.e ** E for E in test_energies]
	rescaled_test_XS = [np.e ** XS for XS in y_test]

	rescaled_predictions = [np.e ** p for p in predictions]

	# mse_loss = mean_squared_error(predictions, y_test)
	mae_loss = mean_absolute_error(rescaled_predictions, rescaled_test_XS)

	return {'loss': mae_loss, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=300,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(50))

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)

with open("best_xgboost_model_500to26k.pkl", "wb") as f:
	pickle.dump(best_model, f)