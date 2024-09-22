import tqdm
import numpy as np
import scipy
from sklearn.preprocessing import minmax_scale

def single_nuclide_make_train(df, val_temperatures=[], test_temperatures=[], minERG=0, maxERG=30e6, use_tqdm = False):
	"""Generates training data matrix"""

	# XS = df['XS']
	# ERG = df['ERG']
	# T = df['T']

	XS_train = []
	ERG_train = []
	T_train = []
	# XS_0K_train = []



	if use_tqdm:
		iterator = tqdm.tqdm(df.iterrows(), total = len(df))
	else:
		iterator = df.iterrows()

	for i, row in iterator:
		if row['T'] in val_temperatures or row['T'] in test_temperatures:
			continue
		if (row['ERG'] * 1e6) > maxERG or (row['ERG'] * 1e6) < minERG:
			continue
		XS_train.append(row['XS'])
		ERG_train.append(row['ERG'])
		T_train.append(row['T'])

	X = np.array([ERG_train, T_train])
	ignore_list = []

	for j_idx, feature_list in enumerate(X):
		if j_idx not in ignore_list:
			X[j_idx] = minmax_scale(feature_list)


	y = np.array(XS_train)
	y = minmax_scale(y)

	X = np.transpose(X)

	return X, y, ERG_train, XS_train



def single_nuclide_make_test(df, test_temperatures = [], use_tqdm = False, minERG = 0, maxERG = 30e6):

	XS_test = []
	ERG_test = []
	unscaled_energy = []
	T_test = []

	T = df['T'].values
	Tscaled = minmax_scale(T)




	if use_tqdm:
		iterator = tqdm.tqdm(df.iterrows(), total = len(df))
	else:
		iterator = df.iterrows()

	for i, row in iterator:
		if (row['ERG'] * 1e6) > maxERG or (row['ERG'] * 1e6) < minERG:
			continue
		if row['T'] in test_temperatures:
			XS_test.append(row['XS'])
			ERG_test.append(row['ERG'])
			T_test.append(Tscaled[i])
			unscaled_energy.append(row['ERG'])
		else:
			continue

	X = np.array([ERG_test, T_test])
	ignore_list = [1]

	for j_idx, feature_list in enumerate(X):
		if j_idx not in ignore_list:
			X[j_idx] = minmax_scale(feature_list)

	y = np.array(XS_test)
	y = minmax_scale(y)

	X = np.transpose(X)



	return X, y, unscaled_energy, XS_test










def single_nuclide_make_val(df, val_temperatures = [], use_tqdm = False, minERG = 0, maxERG = 30e6):

	XS_test = []
	ERG_test = []
	unscaled_energy = []
	T_test = []

	T = df['T'].values
	Tscaled = minmax_scale(T)


	if use_tqdm:
		iterator = tqdm.tqdm(df.iterrows(), total = len(df))
	else:
		iterator = df.iterrows()

	for i, row in iterator:
		if (row['ERG'] * 1e6) > maxERG or (row['ERG'] * 1e6) < minERG:
			continue
		if row['T'] in val_temperatures:
			XS_test.append(row['XS'])
			ERG_test.append(row['ERG'])
			T_test.append(Tscaled[i])
			unscaled_energy.append(row['ERG'])
		else:
			continue

	X = np.array([ERG_test, T_test])
	ignore_list = [1]

	for j_idx, feature_list in enumerate(X):
		if j_idx not in ignore_list:
			X[j_idx] = minmax_scale(feature_list)

	y = np.array(XS_test)
	y = minmax_scale(y)


	X = np.transpose(X)



	return X, y, unscaled_energy








# def scaler(X_matrix):
#
# 	ignore_list = []
#
# 	X_matrix = X_matrix.transpose()
# 	for j_idx, feature_list in enumerate(X_matrix):
# 		if j_idx not in ignore_list:
# 			X_matrix[j_idx] = scipy.stats.zscore(feature_list)
#
# 	means = []
# 	stds = []
# 	for feature in X_matrix:
# 		means.append(np.mean(feature))
# 		stds.append(np.std(feature))