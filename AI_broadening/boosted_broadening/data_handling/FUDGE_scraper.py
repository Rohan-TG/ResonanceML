import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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

def bounds(lowerBound, upperBound, unheatedXS, unheatedERG, heatedXS, heatedERG):

	plotXSHeated = []
	plotERGHeated = []
	for i, j in zip(heatedXS, heatedERG):
		if j < upperBound and j > lowerBound:
			plotXSHeated.append(i)
			plotERGHeated.append(j)

	plotXSunheated = []
	plotERGUnheated = []
	for i, j in zip(unheatedXS, unheatedERG):
		if j < upperBound and j > lowerBound:
			plotXSunheated.append(i)
			plotERGUnheated.append(j)


	plt.figure()
	plt.grid()
	plt.plot(plotERGUnheated, plotXSunheated, label = "0 K")
	plt.plot(plotERGHeated, plotXSHeated, label = "1,800 K")
	plt.yscale('log')
	plt.xscale('log')
	plt.legend()
	plt.xlabel('Energy / eV')
	plt.ylabel('$\sigma_{n,\gamma}$ / b')
	plt.title("Doppler Broadened $\sigma_{n,\gamma}$")
	plt.show()

