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