import pandas as pd
import tqdm
# import dask.dataframe
import os


largecsvfiles = os.listdir('/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT0.1K_csv_samples')


for filename in tqdm.tqdm(largecsvfiles, total=len(largecsvfiles)):
	xs_list = []
	erg_list = []
	t_list = []

	df = pd.read_csv(f'/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT0.1K_csv_samples/{filename}')

	all_temperatures = list(df['T'].values)
	unique_temps = list(set(all_temperatures))

	for T in unique_temps:
		truncateddf = df[df['T'] == T]
		savename = f'Fe_56_{T}K'
		truncateddf.to_csv(f'/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT0.1k_single_temp/{savename}.csv')

