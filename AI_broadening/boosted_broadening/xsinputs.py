import pandas as pd
import numpy as np
import tqdm

# df0 = pd.read_csv('Fe56_MT_102_eV_0K_to_4000K_Delta20K.csv')

dfmain = pd.read_csv('../AI_data/Fe56_200_to_1800_D1K_MT102.csv')


minheated_energies = dfmain[(dfmain['T'] == 200)]['ERG'].values
minheated_XS = dfmain[(dfmain['T'] == 200)]['XS'].values

alltemperatures = np.arange(200, 1801, 1)

matrix = []
alltemperatures = alltemperatures[alltemperatures != 1250]

for temp in tqdm.tqdm(alltemperatures, total=len(alltemperatures)):
	tempXS = dfmain[(dfmain['T'] == temp)]['XS'].values
	tempERG = dfmain[(dfmain['T'] == temp)]['ERG'].values
	interpolated_cross_sections = np.interp(minheated_energies, tempERG, tempXS)
	temperatureSeries = [temp for i in interpolated_cross_sections]

	submatrix = [minheated_energies, interpolated_cross_sections, temperatureSeries]

	matrix.append(submatrix)

matrix = np.array(matrix)

for sample in matrix:
	sampleTemperature = sample[2][0]
