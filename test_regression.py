import xgboost as xg
import pandas as pd
from resml_functions import General_plotter, range_setter
import matplotlib.pyplot as plt
import math
import numpy as np
import tqdm
import periodictable
print('Imports successful...')
df = pd.read_csv('ENDFBVIII_resonance_parameter_trial_1.csv')

eV_levels = []

print('Data loaded')


min_energy = 1
max_energy = 21e6


def train_matrix(df, train_nuclides, val_nuclides, LA, UA, use_tqdm=False):

	Z = df['Z']
	A = df['A']
	Q = df['Q']
	XS = df['XS']
	ERG = df['ERG']
	res_erg = df['res_e']
	Z_train = []
	A_train = []
	Q_train = []
	ERG_train = []
	res_erg_train = []

	XS_train = []

	iterator = tqdm.tqdm(enumerate(Z), total=len(Z)) if use_tqdm == True else enumerate(Z)

	for i, u in iterator:
		if [Z[i], A[i]] in val_nuclides:
			continue
		if [Z[i], A[i]] not in train_nuclides:
			continue
		if A[i] <= UA and A[i] >= LA and ERG[i] > min_energy and ERG[i] < max_energy:
			Z_train.append(Z[i])
			A_train.append(A[i])
			ERG_train.append(ERG[i])
			Q_train.append(Q[i])
			XS_train.append(XS[i])
			res_erg_train.append(res_erg[i])

	X = np.array([Z_train, A_train, Q_train, ERG_train, res_erg_train])

	y = np.array(XS_train)

	X = np.transpose(X)
	return X, y


def test_matrix(df, val_nuclides, ):

	ztest = [nuclide[0] for nuclide in val_nuclides]  # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in val_nuclides]

	Z = df['Z']
	A = df['A']
	Q = df['Q']
	XS = df['XS']
	ERG = df['ERG']
	res_erg = df['res_e']


	Z_test = []
	A_test = []
	Q_test = []
	ERG_test = []
	res_erg_test = []

	XS_test = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and ERG[j] > min_energy and ERG[j] < max_energy:
				Z_test.append(Z[j])
				A_test.append(A[j])
				Q_test.append(Q[j])
				ERG_test.append(ERG[j])
				XS_test.append(XS[j])
				res_erg_test.append(res_erg[j])


	xtest = np.array([Z_test, A_test, Q_test, ERG_test, res_erg_test])

	xtest = np.transpose(xtest)

	y_test = XS_test

	return xtest, y_test












# tempsmall = df[df.A > 10]
# tempsmall2 = tempsmall[tempsmall.A < 60]
# tempsmall2.index = range(len(tempsmall2))
al = range_setter(df=df, la=0, ua=260, use_tqdm=True)

# erg, xs = General_plotter(df=df, nuclides=[[82,208]])
#
target = [82,208]
energy_grid, zrxs = General_plotter(df=df, nuclides=[target])


t_n = [[17,35], [17,36], [14,29], [14,28], [82,207], [20,40], [19,39], [82,206]]

plot_energy_grid = []
plotxs = []

for e, xs in zip(energy_grid, zrxs):
	if e > min_energy:
		plot_energy_grid.append(e)
		plotxs.append(xs)

print('Data loaded. Forming matrices...')

validation_nuclides = [target]

X_train, y_train = train_matrix(df=df,train_nuclides=t_n, val_nuclides=validation_nuclides, LA=0, UA=270,
								use_tqdm=True)

X_test, y_test = test_matrix(df=df, val_nuclides=validation_nuclides)

print('Matrices formed. Training...')

model = xg.XGBRegressor(n_estimators= 800,
						max_depth=6,
						learning_rate=0.1,
						# subsample=0.5,
						max_leaves=0)

model.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

print('Training complete. Evaluating...')

predictions = model.predict(X_test)

logp = [np.log(abs(p)) for p in predictions]
loge = [np.log(e) for e in plot_energy_grid]
logxs = [np.log(x) for x in plotxs]


rps = df[(df['Z'] == target[0]) & (df['A'] == target[1])]['res_e'].values

logrp = []
for i in rps:
	if not math.isnan(i):
		logrp.append(0)
	else:
		logrp.append(np.nan)

logrp = logrp[25001:]
plt.figure()
plt.plot(loge, logxs, label = 'ENDF/B-VIII')
plt.plot(loge,logrp, label = 'Labels')
plt.plot(loge, logp, color='red', label = 'Predictions')
plt.title(f"$\sigma_{{n,\gamma}}$ predictions for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.legend()
plt.grid()
plt.xlabel('Log energy')
plt.ylabel('Log XS')
plt.show()


model.get_booster().feature_names = ['Z', 'A', 'Q', 'ERG', 'res_erg']

plt.figure()
plt.title('Gain')
xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=50)
plt.show()


plt.figure()
plt.title('Splits')
xg.plot_importance(model, ax=plt.gca())
plt.show()