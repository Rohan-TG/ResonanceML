import pandas as pd
import periodictable
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matrix_functions import General_plotter

# df = pd.read_csv('test2.csv',
# 				 # header=2,
# 				 delimiter=';',
# 				 )

df = pd.read_csv('ENDFB8_MT102_log5000.csv')

# df.columns = df.columns.str.split('.').str[0] # allows duplicate column names
# old_column_names = df.columns
#
# new_names = []
# for name in old_column_names:
# 	new_names.append(name.strip())
#
#
# parity_names = []
# final_headers = []
# for name in new_names:
# 	if name in parity_names:
# 		delta_header = name + ' ds'
# 		final_headers.append(delta_header)
# 	else:
# 		final_headers.append(name)
# 		parity_names.append(name)
#
# df2 = df.set_axis(final_headers, axis=1)
#
def nuclide_extractor(head):
	element_name = ''
	isotope_string = ''
	for char in head:
		if char == ' ':
			break
		try:
			integer = int(char)
			isotope_string += char
		except ValueError:
			element_name += char

	isotope = int(isotope_string)
	atomic_number = periodictable.elements.symbol(element_name).number
	return(atomic_number, isotope)
#
# for header in df2.columns[1:]:
# 	nuclide_extractor(head=header)
#
# mt102 = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])
#
ERG = []
Z = []
A = []
XS = []
# dXS = []
# for i, row in tqdm.tqdm(df2.iterrows(), total=df2.shape[0]):
# 	# print(f"{i}/{len(df2)}")
# 	if i > 1:
#
# 		parity_bits = [1]
#
# 		for name in final_headers[1:]:
# 			if 'ds' not in name:
# 				z, a = nuclide_extractor(head=name)
# 				Z.append(z)
# 				A.append(a)
#
#
#
# 				if type(row['Incident energy']) == str:
# 					energy = float(row['Incident energy'].strip())
# 					ERG.append(energy)
# 				elif type(row['Incident energy']) == float:
# 					# print(row['Incident energy'])
# 					ERG.append(row['Incident energy'])
#
# 				if parity_bits[-1] == 0:
# 					dXS.append(np.nan)
#
# 				if type(row[name]) == str:
# 					try:
# 						xs_value = float(row[name].strip())
# 					except ValueError:
# 						xs_value = np.nan
# 					XS.append(xs_value)
# 				elif type(row[name]) == float:
# 					XS.append(row[name])
# 				parity_bits.append(0)
# 			if 'ds' in name:
# 				parity_bits.append(1)
# 				if type(row[name]) == str:
# 					try:
# 						dxs_value = float(row[name].strip())
# 					except ValueError:
# 						dxs_value = np.nan
# 					dXS.append(dxs_value)
# 				elif type(row[name]) == float:
# 					dXS.append(row[name])
#
# temp = pd.DataFrame({'Z': Z,
# 					 'A': A,
# 					 'ERG': ERG,
# 					 'XS': XS})
#
# temptarget = [74,184]
# erg, xs = General_plotter(df=temp, nuclides=[temptarget])
# logerg = [np.log10(i) for i in erg]
# logxs = [np.log10(i) for i in xs]
#
# plt.figure()
# plt.plot(logerg, logxs)
# plt.xlabel('Energy')
# plt.ylabel('b')
# plt.title(f"$\sigma_{{n,\gamma}}$ for {periodictable.elements[temptarget[0]]}-{temptarget[1]}")
# plt.grid()
# plt.show()
mt102df = pd.DataFrame(columns=['Z', 'A', 'N', 'ERG', 'XS'])

full_energy_list = []
full_z_list = []
full_a_list = []
full_xs_list = []

for species in tqdm.tqdm(df.columns[1:], total=len(df.columns[1:])):
	if 'm' not in species:
		tempseries = df[species]
		tempz, tempa = nuclide_extractor(head=species)
		tempn = tempa - tempz
		for species_xs, erg_i in zip(tempseries, df['E']):
			full_a_list.append(tempa)
			full_z_list.append(tempz)
			full_xs_list.append(species_xs)
			full_energy_list.append(erg_i)