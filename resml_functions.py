import numpy as np
import tqdm
import pandas as pd

def General_plotter(df, nuclides):
	"""df: dataframe source of XSs
	nuclides: must be array of 1x2 arrays [z,a]

	Returns XS and ERG values. Designed for plotting graphs and doing r2 comparisons without running make_test
	which is much more demanding"""

	ztest = [nuclide[0] for nuclide in nuclides]  # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in nuclides]

	Z = df['Z']
	A = df['A']

	Energy = df['ERG']
	XS = df['XS']


	Z_test = []
	A_test = []
	Energy_test = []
	XS_test = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a:
				Z_test.append(Z[j])
				A_test.append(A[j])
				Energy_test.append(Energy[j])
				XS_test.append(XS[j])

	energies = np.array(Energy_test)

	xs = XS_test

	return energies, xs




def range_setter(df, la, ua):
	nucs = []

	for i, j in zip(df['Z'], df['A']):
		if [i, j] in nucs or j > ua or j < la:
			continue
		else:
			nucs.append([i, j])

	return nucs


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
			c_levels_train.append(c_levels[i])

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
				c_levels_test.append(c_levels[j])


	xtest = np.array([Z_test, A_test, Q_test, ERG_test, c_levels_test])

	xtest = np.transpose(xtest)

	y_test = XS_test

	return xtest, y_test