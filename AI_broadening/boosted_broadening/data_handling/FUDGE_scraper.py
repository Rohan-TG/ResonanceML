import pandas as pd
import os
import numpy as np
import tqdm

xml_folder = '/Users/rntg/PycharmProjects/ResonanceML/fudge/broadening_tests/n+Fe56/heated'

heated_files = os.listdir(xml_folder)
# heated_files.remove('.DS_Store') # Only necessary if on Mac

heated_files.sort()

T_list = []
xs_list = []
erg_list = []

file_temps = np.arange(0, 1900, 100)

for folder, folder_temperature in tqdm.tqdm(zip(heated_files, file_temps), total=len(heated_files)):
	file = np.loadtxt(f'{xml_folder}/{folder}/063_102.dat')
	for pair in file:
		erg_list.append(pair[0])
		xs_list.append(pair[1])
		T_list.append(folder_temperature)

df = pd.DataFrame({'ERG': erg_list, 'XS': xs_list, 'T': T_list})

# df.to_csv('')