import numpy as np
import tqdm


def single_nuclide_make_train(df, val_temperatures=[], test_temperatures=[], minERG=0, maxERG=30e6, use_tqdm = False):
	"""Generates training data matrix"""

	# XS = df['XS']
	# ERG = df['ERG']
	# T = df['T']

	XS_train = []
	ERG_train = []
	T_train = []
	XS_0K_train = []



	if use_tqdm:
		iterator = tqdm.tqdm(df.iterrows(), total = len(df))
	else:
		iterator = df.iterrows()

	for i, row in iterator:
		if row['T'] in val_temperatures or row['T'] in test_temperatures:
			continue
		if row['ERG'] > maxERG or row['ERG'] < minERG:
			continue
		XS_train.append(row['XS'])
		ERG_train.append(row['ERG'])
		T_train.append(row['T'])

	X = np.array([ERG_train, T_train])

	y = np.array(XS_train)

	X = np.transpose(X)

	return X, y


def single_nuclide_make_test(df, test_temperatures = [], use_tqdm = False, minERG = 0, maxERG = 30e6):

	XS_test = []
	ERG_test = []
	T_test = []

	DF = df[(df['T'] == test_temperatures[0])]
	DF.index = range(len(DF))

	if use_tqdm:
		iterator = tqdm.tqdm(DF.iterrows(), total = len(DF))
	else:
		iterator = DF.iterrows()

	for i, row in iterator:
		if row['ERG'] > maxERG or row['ERG'] < minERG:
			continue
		if row['T'] in test_temperatures:
			XS_test.append(row['XS'])
			ERG_test.append(row['ERG'])
			T_test.append(row['T'])
		else:
			continue

	X = np.array([ERG_test, T_test])

	y = np.array(XS_test)

	X = np.transpose(X)

	return X, y





def log_single_nuclide_data_maker(df, val_temperatures = [], test_temperatures = [], use_tqdm = False,
							  minERG = 0, maxERG = 30e6):

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
		if row['ERG'] > maxERG or row['ERG'] < minERG:
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

	X_train = np.array([np.log(ERG_train), T_train])
	y_train = np.array(np.log(XS_train))
	X_train = np.transpose(X_train)


	X_val = np.array([np.log(ERG_val), T_val])
	y_val = np.array(np.log(XS_val))
	X_val = np.transpose(X_val)


	X_test = np.array([np.log(ERG_test), T_test])
	y_test = np.array(np.log(XS_test))
	X_test = np.transpose(X_test)

	return X_train, y_train, X_val, y_val, X_test, y_test























def single_nuclide_data_maker(df, val_temperatures = [], test_temperatures = [], use_tqdm = False,
							  minERG = 0, maxERG = 30e6):

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
		if row['ERG'] > maxERG or row['ERG'] < minERG:
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
	X_train = np.transpose(X_train)


	X_val = np.array([ERG_val, T_val])
	y_val = np.array(XS_val)
	X_val = np.transpose(X_val)


	X_test = np.array([ERG_test, T_test])
	y_test = np.array(XS_test)
	X_test = np.transpose(X_test)

	return X_train, y_train, X_val, y_val, X_test, y_test