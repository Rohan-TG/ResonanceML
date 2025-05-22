import os

os.environ["OMP_NUM_THREADS"] = "30"
os.environ["MKL_NUM_THREADS"] = "30"
os.environ["OPENBLAS_NUM_THREADS"] = "30"
os.environ["TF_NUM_INTEROP_THREADS"] = "30"
os.environ["TF_NUM_INTRAOP_THREADS"] = "30"

import pandas as pd
import keras
import datetime
import numpy as np
import random
import periodictable
import itertools
import scipy
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import mean_absolute_error

def get_datetime_string():
	return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

maxtemp = 1200
mintemp = 1000

minerg = 1000/ 1e6
maxerg = 1200 / 1e6


numbers = np.linspace(mintemp, maxtemp, int((maxtemp - mintemp) / 0.1) + 1, dtype=np.float32) # all temperatures in the data file
all_temperatures = [round(NUM, 1) for NUM in numbers]

test_temperatures = [1300.0]

data_dir = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/interpolated_high_res_0.1K'

df0 = pd.read_csv('../AI_data/Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')
unheated_energies = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg * 1e6)) & (df0['ERG'] < (maxerg * 1e6))]['ERG'].values
unheated_XS = df0[(df0['T'] == 0) & (df0['ERG'] > (minerg * 1e6)) & (df0['ERG'] < (maxerg * 1e6))]['XS'].values

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


filenames = os.listdir(data_dir)
exclusions = [254.7, 254.8, 254.9, 255.0]


train_input_matrix = [] # contains the energy grid and temperatures
train_labels_matrix = [] # contains the cross sections

for train_temperature in tqdm.tqdm(training_temperatures, total = len(training_temperatures)):
	if round(float(train_temperature), 1) not in exclusions:
		roundedtt = str(round(train_temperature, 1))
		filename = f'Fe_56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		logged_T_values = np.log10(df['T'].values)
		scaled_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_T_values]

		input_submatrix = np.array(scaled_T_values)  # can add or remove ERG here to make energy an input parameter
		labelsubmatrix = np.array(np.log10(df['XS'].values))

		train_input_matrix.append(input_submatrix)
		train_labels_matrix.append(labelsubmatrix)

X_train = np.array(train_input_matrix)
y_train = np.array(train_labels_matrix)

flattened_y_train = y_train.flatten()
meanXS_train = np.mean(flattened_y_train)
stdXS_train = np.std(flattened_y_train)

scaled_train_labels_matrix = []
for set in train_labels_matrix:
	scaled_set = [(xs - meanXS_train) / stdXS_train for xs in set]
	scaled_train_labels_matrix.append(scaled_set)

y_train = np.array(scaled_train_labels_matrix)


val_input_matrix = [] # contains the energy grid and temperatures
val_labels_matrix = [] # contains the cross sections

for val_temperature in tqdm.tqdm(validation_temperatures, total=len(validation_temperatures)):
	if round(float(val_temperature), 1) not in exclusions:
		roundedtt = str(round(val_temperature, 1))
		filename = f'Fe_56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		logged_val_T_values = np.log10(df['T'].values)
		scaled_val_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_val_T_values]

		input_submatrix_val = np.array(scaled_val_T_values)  # can add or remove ERG here to make energy an input parameter
		labelsubmatrix_val = np.array(np.log10(df['XS'].values))

		val_input_matrix.append(input_submatrix_val)
		val_labels_matrix.append(labelsubmatrix_val)

X_val = np.array(val_input_matrix)
y_val = np.array(val_labels_matrix)

flattened_y_val = y_val.flatten()
meanXS_val = np.mean(flattened_y_val)
stdXS_val = np.std(flattened_y_val)

scaled_val_labels_matrix = []
for set in val_labels_matrix:
	scaled_set = [(xs - meanXS_val) / stdXS_val for xs in set]
	scaled_val_labels_matrix.append(scaled_set)

y_val = np.array(scaled_val_labels_matrix)



test_input_matrix = []
test_labels_matrix = []

for test_temperature in tqdm.tqdm(test_temperatures, total=len(test_temperatures)):
	if round(float(test_temperature), 1) not in exclusions:
		roundedtt = str(round(test_temperature, 1))
		filename = f'Fe_56_{roundedtt}.csv'
		df = pd.read_csv(f'{data_dir}/{filename}')
		df = df[(df['ERG'] < maxerg) & (df['ERG'] > minerg)]

		logged_test_T_values = np.log10(df['T'].values)
		scaled_test_T_values = [(t - mean_alltemps) / std_alltemps for t in logged_test_T_values]

		input_submatrix_test = np.array(scaled_test_T_values)  # can add or remove ERG here to make energy an input parameter
		labelsubmatrix_test = np.array(np.log10(df['XS'].values))

		test_input_matrix.append(input_submatrix_test)
		test_labels_matrix.append(labelsubmatrix_test)

X_test = np.array(test_input_matrix)
y_test = np.array(test_labels_matrix)

flattened_y_test = y_test.flatten()
meanXS_test = np.mean(flattened_y_test)
stdXS_test = np.std(flattened_y_test)

scaled_test_labels_matrix = []
for set in test_labels_matrix:
	scaled_set = [(xs - meanXS_test) / stdXS_test for xs in set]
	scaled_test_labels_matrix.append(scaled_set)

y_test = np.array(scaled_test_labels_matrix)