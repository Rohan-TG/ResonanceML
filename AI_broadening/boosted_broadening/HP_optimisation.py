# Bayesian HP optimisation script
import random
from funcs import single_nuclide_data_maker
import xgboost as xg
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import hyperopt.early_stop
from hyperopt import  hp, fmin, tpe, STATUS_OK, Trials

df = pd.read_csv('Fe56_MT_102_Delta50K_0K_1800K.csv')


ntreeguess = np.arange(500, 10000, 100)
depthguess = [3,4,5,6,7,8,9,10,11,12]

space = {'n_estimators': hp.choice('n_estimators', ntreeguess),
		 'subsample': hp.uniform('subsample', 0.01, 0.99),
		 'max_leaves': 0,
		 'max_depth': hp.choice('max_depth', depthguess),
		 'reg_lambda': hp.uniform('reg_lambda', 0, 100),
		 'learning_rate': hp.uniform('learning_rate', 0.0001, 0.2)}


nuclide = [26, 56]
minerg = 0 # in eV
maxerg = 1 * 2.9e4 # in eV

all_temperatures = [   0,   50,  100,  150,  200,  250,  300,  350,  400,  450,  500,
        550,  600,  650,  700,  750,  800,  850,  900,  950, 1000, 1050,
       1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600,
       1650, 1700, 1750, 1800]
def optimiser(space):
	test_temperatures = [random.choice(all_temperatures)]
	validation_temperatures = []
	while len(validation_temperatures) < 2:
		valchoice = random.choice(all_temperatures)
		if valchoice not in test_temperatures:
			validation_temperatures.append(valchoice)

	X_train, y_train, X_val, y_val, X_test, y_test = single_nuclide_data_maker(df=df,
												 val_temperatures=validation_temperatures,
												 test_temperatures=test_temperatures,
												 minERG=minerg,
												 maxERG=maxerg,
												 use_tqdm=False)


	model = xg.XGBRegressor(**space, seed=42)

	evaluation = [(X_train, y_train), (X_val, y_val)]
	model.fit(X_train, y_train,
			  eval_set=evaluation,
			  eval_metric='rmse',
			  verbose=True)

	predictions = model.predict(X_test)
	history = model.evals_result()

	test_energies = X_test.transpose()[0]
	# test_energies = [e * 1e6 for e in test_energies]

	unheated_energies = df[(df['T'] == 0) & (df['ERG'] > (minerg)) & (df['ERG'] < (maxerg))]['ERG'].values
	unheated_energies = [e for e in unheated_energies]
	unheated_XS = df[(df['T'] == 0) & (df['ERG'] > (minerg)) & (df['ERG'] < (maxerg))]['XS'].values


	mse_loss = mean_squared_error(predictions, y_test)

	return {'loss': mse_loss, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=100,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(50))

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)