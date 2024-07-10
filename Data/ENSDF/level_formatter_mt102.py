import pandas as pd

from nudel.nudel import Nuclide
import resml_functions
import numpy as np
import tqdm

fe56 = Nuclide(56,26)

print(fe56)



fe56_levels = fe56.adopted_levels.levels

fe56_level1 = fe56.adopted_levels.levels[1]

fe56_level1_energy = fe56_level1.energy.val

df = pd.read_csv('ENDFBVIII_MT102_XS_only.csv')
energy_grid, unused = resml_functions.General_plotter(df=df, nuclides = [[26,56]])

ENDFB_nuclides = resml_functions.range_setter(df=df, la=0, ua=260)

print('Data loaded')

full_level_energy_array = []
full_level_energy_error_array = []
ensdf_nuclide_list = []

for endfb_nuclide in ENDFB_nuclides: # for each nuclide in endfb8
	try:
		temp_nuclide = Nuclide(endfb_nuclide[1], endfb_nuclide[0]) # extract ENSDF data for the nuclide

		nuclide_levels = temp_nuclide.adopted_levels.levels # object containing the level structure and associated data

		float_nuclide_level_energies = [] # list of the level energies for the endfb8 nuclide in question
		float_level_error = []

		for l in nuclide_levels:
			level_energy = l.energy.val
			MeV_level_energy = level_energy / 1000

			float_level_error.append(l.energy.pm / 1000)
			float_nuclide_level_energies.append(MeV_level_energy) # converts to MeV

		full_level_energy_array.append(float_nuclide_level_energies)
		full_level_energy_error_array.append(float_level_error)
		ns = temp_nuclide.mass - temp_nuclide.protons
		ensdf_nuclide_list.append([temp_nuclide.protons, ns])
	except:
		print(endfb_nuclide, 'not in ENSDF')


final_level_energies = [] #will contain all values, including nans, for level energies across all nuclides
# for i, row in df.iterrows():
# 	row_nuclide = [row['Z'], row['A']]
# 	if row_nuclide in ensdf_nuclide_list:
# 		fetch_index = ensdf_nuclide_list.index(row_nuclide)
#
trial = []
counter = 0
for e in tqdm.tqdm(energy_grid, total=len(energy_grid)):
	if counter > (len(float_nuclide_level_energies) - 1):
		trial.append(np.nan)
	elif e < float_nuclide_level_energies[counter]:
		trial.append(np.nan)
	else:
		trial.append(float_nuclide_level_energies[counter])
		counter += 1