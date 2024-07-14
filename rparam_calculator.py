import numpy as np
import pandas as pd
import math
from resml_functions import General_plotter, range_setter
import tqdm
print('Imports successful...')

q_data = pd.read_csv('ENDFBVIII_MT102_Q_values.csv')
print('Q values loaded')
df = pd.read_csv('ENDFBVIII_MT102_levels.csv')
print('Mainframe loaded')

energy_grid, unused_xs = General_plotter(df=df, nuclides=[[17,35]])

endfbnuclides = range_setter(df=df, la=0, ua=260)
c_levels = df['c_levels']


def closest_energy(neutron_energies, level_energy_value):
	closest_index = min(range((len(neutron_energies))), key=lambda i: abs(neutron_energies[i]-level_energy_value))
	closest_value = neutron_energies[closest_index]
	return(closest_value, closest_index)

total_res_param_list = []

for nuclide in endfbnuclides:
	minidf= df[(df['Z'] == nuclide[0]) & (df['A'] == nuclide[1])]
	minidf.index = range(len(minidf))
	pass
resonance_energy = np.zeros(len(energy_grid))
for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
	if not math.isnan(row['c_levels']) and row['c_levels'] != 0:

		qvalue = q_data[(q_data['Z'] == row['Z']) & (q_data['A'] == row['A'])]['Q']
#
		compound_level = row['c_levels']

		predicted_resonant_energy_pdtype = compound_level - qvalue
		predicted_resonant_energy = predicted_resonant_energy_pdtype.values[0]
#
		if predicted_resonant_energy < 0:
			continue
		else:
			fill = range(i-100, i+100)
			for j in fill:
				match_neutron_energy = closest_energy(neutron_energies=energy_grid,
													  level_energy_value=predicted_resonant_energy)
