import numpy as np
import pandas as pd
import math
import tqdm
print('Imports successful...')

q_data = pd.read_csv('ENDFBVIII_MT102_Q_values.csv')
print('Q values loaded')
df = pd.read_csv('ENDFBVIII_MT102_levels.csv')
print('Mainframe loaded')

resonance_energy = []

c_levels = df['c_levels']

minidf = df[df.Z == 17]
minidf = minidf[minidf.A == 35]
minidf.index = range(len(minidf))

for i, row in tqdm.tqdm(minidf.iterrows(), total=minidf.shape[0]):
	if not math.isnan(row['c_levels']) and row['c_levels'] != 0:

		qvalue = q_data[(q_data['Z'] == row['Z']) & (q_data['A'] == row['A'])]['Q']

		compound_level = row['c_levels'] * 1e6 # convert to eV from MeV

		predicted_resonant_energy_pdtype = compound_level - qvalue

		predicted_resonant_energy = predicted_resonant_energy_pdtype.values[0]

		if predicted_resonant_energy < 0:
			resonance_energy.append(np.nan)
		else:
			resonance_energy.append(predicted_resonant_energy)

	else:
		resonance_energy.append(np.nan)