import tqdm
import numpy as np
import scipy
# from sklearn.preprocessing import minmax_scale

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
			X[j_idx] = scipy.stats.zscore(feature_list)


	y = np.array(XS_train)
	y = scipy.stats.zscore(y)

	X = np.transpose(X)

	return X, y, ERG_train, XS_train



def single_nuclide_make_test(df, test_temperatures = [], use_tqdm = False, minERG = 0, maxERG = 30e6):

	XS_test = []
	ERG_test = []
	unscaled_energy = []
	T_test = []

	T = df['T'].values
	Tscaled = scipy.stats.zscore(T)




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
			X[j_idx] = scipy.stats.zscore(feature_list)

	y = np.array(XS_test)
	y = scipy.stats.zscore(y)

	X = np.transpose(X)



	return X, y, unscaled_energy, XS_test









def single_nuclide_make_val(df, val_temperatures = [], use_tqdm = False, minERG = 0, maxERG = 30e6):

	XS_test = []
	ERG_test = []
	unscaled_energy = []
	T_test = []

	T = df['T'].values
	Tscaled = scipy.stats.zscore(T)


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
			X[j_idx] = scipy.stats.zscore(feature_list)

	y = np.array(XS_test)
	y = scipy.stats.zscore(y)


	X = np.transpose(X)



	return X, y, unscaled_energy




def single_nuclide_data_maker(df, val_temperatures = [], test_temperatures = [], use_tqdm = False, minERG = 0, maxERG = 30e6):

	XS_train = []
	ERG_train = []
	T_train = []

	XS_val = []
	ERG_val =[]
	T_val = []

	XS_test = []
	ERG_test = []
	T_test = []

	if use_tqdm:
		iterator = tqdm.tqdm(df.iterrows(), total = len(df))
	else:
		iterator = df.iterrows()

	for i, row in iterator:
		if (row['ERG'] * 1e6) > maxERG or (row['ERG'] * 1e6) < minERG:
			continue
		if row['T'] in val_temperatures:
			XS_val.append(row['XS'])
			ERG_val.append(row['ERG'])
			T_val.append(row['T'])
		if row['T'] in test_temperatures:
			XS_test.append(row['XS'])
			ERG_test.append(row['ERG'])
			T_test.append(row['T'])
		if row['T'] not in val_temperatures and row['T'] not in test_temperatures:
			XS_train.append(row['XS'])
			ERG_train.append(row['ERG'])
			T_train.append(row['T'])

	X_train = np.array([ERG_train, T_train])
	y_train = np.array(XS_train)
	y_train = scipy.stats.zscore(y_train)

	feature_means = []
	feature_stds = []
	for j_idx, feature_list in enumerate(X_train):
		X_train[j_idx] = scipy.stats.zscore(feature_list)
		feature_means.append(np.mean(feature_list))
		feature_stds.append(np.std(feature_list))

	X_train = np.transpose(X_train)

	# ERG_train_mean = np.mean(ERG_train)
	# ERG_train_std = np.std(ERG_train)
	#
	# T_train_mean = np.mean(T_train)
	# T_train_std = np.std(T_train)
	#
	# XS_train_mean = np.mean(XS_train)
	# XS_train_std = np.std(XS_train)

	X_val = np.array([ERG_val, T_val])
	y_val = np.array([XS_val])
	y_val = scipy.stats.zscore(y_val)
	for idx, (feature_list, f_mean, f_std) in enumerate(zip(X_val, feature_means, feature_stds)):
		X_val[idx] = (feature_list - f_mean) / f_std

	X_val = np.transpose(X_val)

	X_test= np.array([ERG_test, T_test])
	for idx, (feature_list, f_mean, f_std) in enumerate(zip(X_test, feature_means, feature_stds)):
		X_test[idx] = (feature_list - f_mean) / f_std

	X_test = np.transpose(X_test)
	y_test= np.array([XS_test])
	y_test = scipy.stats.zscore(y_test)



	return X_train, y_train, ERG_train, XS_train, X_val, y_val, ERG_val, XS_val, X_test, y_test, ERG_test, XS_test



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