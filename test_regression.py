import xgboost as xg
import pandas as pd
from resml_functions import General_plotter
import matplotlib.pyplot as plt
import math
import numpy as np
import tqdm
import periodictable
print('Imports successful...')
df = pd.read_csv('ENDFBVIII_MT102_levels.csv')

eV_levels = []


for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
	cval = row['c_levels']
	if not math.isnan(cval):

		eV_cval = cval * 1e6
		eV_levels.append(eV_cval)
		print(cval)
		break
	else:
		eV_levels.append(np.nan)





def train_matrix(df, val_nuclides, LA, UA):

	Z = df['Z']
	A = df['A']
	Q = df['Q']
	XS = df['XS']
	ERG = df['ERG']
	c_levels = df['c_levels']
	print(c_levels)
	print(XS)
	Z_train = []
	A_train = []
	Q_train = []
	ERG_train = []
	c_levels_train = []

	XS_train = []

	for i, u in tqdm.tqdm(enumerate(Z), total=len(Z)):
		if [Z[i], A[i]] in val_nuclides:
			continue
		if A[i] <= UA and A[i] >= LA:
			Z_train.append(Z[i])
			A_train.append(A[i])
			ERG_train.append(ERG[i])
			Q_train.append(Q[i])
			XS_train.append(XS[i])
			c_levels_train.append(c_levels[i]*1e6)

	X = np.array([Z_train, A_train, Q_train, ERG_train, c_levels_train])

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
	c_levels = df['c_levels']


	Z_test = []
	A_test = []
	Q_test = []
	ERG_test = []
	c_levels_test = []

	XS_test = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a:
				Z_test.append(Z[j])
				A_test.append(A[j])
				Q_test.append(Q[j])
				ERG_test.append(ERG[j])
				XS_test.append(XS[j])
				c_levels_test.append(c_levels[j] * 1e6)


	xtest = np.array([Z_test, A_test, Q_test, ERG_test, c_levels_test])

	xtest = np.transpose(xtest)

	y_test = XS_test

	return xtest, y_test
















# al = resml_functions.range_setter(df=df, la=38, ua=42)

# tempsmall = df[df.A > 20]
# tempsmall2 = tempsmall[tempsmall.A < 80]
# tempsmall2.index = range(len(tempsmall2))
#
# energy_grid, zrxs = General_plotter(df=tempsmall2, nuclides=[[17,35]])
#
# print('Data loaded. Forming matrices...')
#
# validation_nuclides = [[17,35]]
#
# X_train, y_train = train_matrix(df=tempsmall2, val_nuclides=validation_nuclides, LA=20, UA=80)
#
# X_test, y_test = test_matrix(df=tempsmall2, val_nuclides=validation_nuclides)
#
# print('Matrices formed. Training...')
#
# model = xg.XGBRegressor(n_estimators= 500,
# 						max_depth=6,
# 						learning_rate=0.1,
# 						# subsample=0.5,
# 						max_leaves=0)

# model.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])
#
# print('Training complete. Evaluating...')
#
# predictions = model.predict(X_test)
#
# logp = [np.log(abs(p)) for p in predictions]
# loge = [np.log(e) for e in energy_grid]
# logxs = [np.log(x) for x in zrxs]
#
# plt.figure()
# plt.plot(loge, logxs, label = 'ENDF/B-VIII')
# plt.plot(loge, logp, color='red', label = 'Predictions')
# plt.title(f"$\sigma_{{n,\gamma}} predictions for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}$")
# plt.legend()
# plt.grid()
# plt.xlabel('Log energy')
# plt.ylabel('Log XS')
# plt.show()
#
#
# model.get_booster().feature_names = ['Z', 'A', 'Q', 'ERG', 'c_levels']
#
# plt.figure()
# plt.title('Gain')
# xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=50)
# plt.show()
#
#
# plt.figure()
# plt.title('Splits')
# xg.plot_importance(model, ax=plt.gca())
# plt.show()