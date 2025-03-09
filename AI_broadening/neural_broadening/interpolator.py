import os
import pandas as pd
import tqdm
import numpy as np


raw_data_directory = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/dT0.1k_single_temp'

dir_interpolated = '/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/AI_data/interpolated_high_res_0.1K'


raw_files = os.listdir(raw_data_directory)

main_grid_file =  pd.read_csv(f'{raw_data_directory}/Fe_56_200.0K.csv')
main_grid = main_grid_file['ERG'].values

for rawfile in tqdm.tqdm(raw_files, total=len(raw_files)):
	df = pd.read_csv(f'{raw_data_directory}/{rawfile}')
	xs = df['XS'].values
	erg = df['ERG'].values

	new_xs = np.interp(main_grid, erg, xs)
	tlist = [df['T'].values[0] for X in new_xs]

	newdf = pd.DataFrame({'ERG': main_grid, 'XS': new_xs, 'T':tlist})
	newdf.to_csv(f'{dir_interpolated}/Fe_56_{tlist[0]}.csv')