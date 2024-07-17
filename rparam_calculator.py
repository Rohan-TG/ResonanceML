import numpy as np
import pandas as pd
from pygments.lexers import math
import math
from resml_functions import General_plotter, range_setter
import tqdm
print('Imports successful...')



def closest_energy(neutron_energies, search_energy_value):
	'''Returns closest energy value to search_energy_value.
	Returns index of value.'''
	closest_index = min(range((len(neutron_energies))), key=lambda i: abs(neutron_energies[i]-search_energy_value))
	closest_value = neutron_energies[closest_index]
	return(closest_value, closest_index)


df = pd.read_csv('ENDFBVIII_MT102_XS_Q_ZA.csv')
print('Mainframe loaded')


endfbnuclides = range_setter(df=df, la=0, ua=260, use_tqdm=True)
compound_levels = pd.read_csv('ENDFBVIII_compound_level_data.csv')

param_lbound = 30
param_ubound = 100



print('Beginning iteration')
#
# for nuclide in tqdm.tqdm(endfbnuclides, total=len(endfbnuclides)):
energy_grid, unused_xs = General_plotter(df=df, nuclides=[[27,59]])

full_resparam_list = []

for nuclide in tqdm.tqdm(endfbnuclides, total=len(endfbnuclides)):

	dummy_list = [np.nan for i in energy_grid]

	nucdf = df[(df['Z'] == nuclide[0]) & (df['A'] == nuclide[1])]
	nucdf.index = range(len(nucdf))
	qvalue = nucdf['Q'].values[0]

	nucleveldf = compound_levels[(compound_levels['Z'] == nuclide[0]) & (compound_levels['A'] == nuclide[1])]
	nucleveldf.index = range(len(nucleveldf))


	for i, row in nucleveldf.iterrows():
		lc = row['c_levels'] # compound energy level

		predicted_resonance_energy = lc - qvalue

		if predicted_resonance_energy >= 0:

			match_grid_energy, match_index = closest_energy(energy_grid,
															search_energy_value=predicted_resonance_energy)

			fill_indices = range(match_index-param_lbound, match_index+param_ubound)
			if match_index+param_ubound > 61506:
				fill_indices = range(match_index+param_lbound, 61506)
			for r in fill_indices:
				dummy_list[r] = predicted_resonance_energy


	for rp in dummy_list:
		full_resparam_list.append(rp)




# resonance_energy = np.zeros(len(energy_grid))
# for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
# 	if not math.isnan(row['c_levels']) and row['c_levels'] != 0:
#
# 		qvalue = q_data[(q_data['Z'] == row['Z']) & (q_data['A'] == row['A'])]['Q']
# #
# 		compound_level = row['c_levels']
#
# 		predicted_resonant_energy_pdtype = compound_level - qvalue
# 		predicted_resonant_energy = predicted_resonant_energy_pdtype.values[0]
# #
# 		if predicted_resonant_energy < 0:
# 			continue
# 		else:
# 			fill = range(i-100, i+100)
# 			for j in fill:
# 				match_neutron_energy = closest_energy(neutron_energies=energy_grid,
# 													  level_energy_value=predicted_resonant_energy)
