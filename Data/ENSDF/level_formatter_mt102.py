import pandas as pd

from nudel.nudel import Nuclide
import matplotlib.pyplot as plt
import resml_functions
import numpy as np
# import traceback
import tqdm

### NOTE
# This script is obsolete as it produces an outdated level representation. New
# script is called level_data_formatter.py











df = pd.read_csv('ENDFBVIII_MT102_XS_with_QZA.csv')
energy_grid, unused = resml_functions.General_plotter(df=df, nuclides = [[17,35]])

ENDFB_nuclides = resml_functions.range_setter(df=df, la=0, ua=260)

print('Data loaded')

full_level_energy_array = []
full_level_energy_error_array = []
ensdf_nuclide_list = []

# eV_compound_list = []

full_compound_level_list = []
full_compound_level_error_list = []

full_energies_list = []
full_xs_list = []
full_Z_list = []
full_A_list = []
full_q_list = df['Q'].values




missing_nudel = []
for n in ENDFB_nuclides:
	try:
		Nuclide(n[1],n[0])
	except:
		missing_nudel.append(n)


missing_compounds = []
for n in ENDFB_nuclides:
	try:
		Nuclide(n[1]+1,n[0])
	except:
		missing_compounds.append([n[0],n[1]+1])

# ENDFB_nuclides = [[17,35], [17,36]]


def logger(energy, xs):

	logerg = [np.log10(erg) for erg in energy]
	logxs = [np.log10(xval) for xval in xs]

	plt.figure()
	plt.plot(logerg, logxs)
	plt.xlabel('lg(E)')
	plt.ylabel('lg($\sigma_{n,\gamma}$)')
	plt.grid()
	plt.show()

for endfb_nuclide in tqdm.tqdm(ENDFB_nuclides, total=len(ENDFB_nuclides)): # for each nuclide in endfb8
	comp = [endfb_nuclide[0], endfb_nuclide[1] + 1]

	temperg, xs = resml_functions.General_plotter(df=df, nuclides=[endfb_nuclide])
	for XS in xs:
		full_xs_list.append(XS)



	for erg in energy_grid:
		full_energies_list.append(erg)
		full_Z_list.append(endfb_nuclide[0])
		full_A_list.append(endfb_nuclide[1])
	if comp not in missing_compounds: # compound nucleus levels

		compound_nuclide = Nuclide(comp[1], comp[0])

		compound_level_structure = compound_nuclide.adopted_levels.levels
		float_comp_level_energies = []
		float_comp_level_errors = []

		# float_eV_comp = []

		for lc in compound_level_structure: # loop gives the levels of the compound nucleus in MeV
			lc_energy = lc.energy.val
			eV_comp_level_energy = lc_energy * 1e3 # in eV

			# float_comp_level_errors.append(lc.energy.pm /1000)
			float_comp_level_energies.append(eV_comp_level_energy)
			# float_eV_comp.append(eV_comp_level_energy)

		compound_grid_levels = []
		c_counter = 0
		for e in energy_grid:
			if c_counter > (len(float_comp_level_energies) - 1):
				compound_grid_levels.append(np.nan)
			elif e < float_comp_level_energies[c_counter]:
				compound_grid_levels.append(np.nan)
			else:
				compound_grid_levels.append(float_comp_level_energies[c_counter])
				c_counter +=1

		for ce in compound_grid_levels:
			full_compound_level_list.append(ce)
	else:
		for ex in energy_grid:
			full_compound_level_list.append(np.nan)
		print('Error with compound data', endfb_nuclide)




	if endfb_nuclide not in missing_nudel: # target nucleus levels

		temp_nuclide = Nuclide(endfb_nuclide[1], endfb_nuclide[0])  # extract ENSDF data for the nuclide

		nuclide_levels = temp_nuclide.adopted_levels.levels # object containing the level structure and associated data

		float_nuclide_level_energies = [] # list of the level energies for the endfb8 nuclide in question
		float_level_error = []

		for l in nuclide_levels:
			level_energy = l.energy.val
			MeV_level_energy = level_energy * 1e3

			float_level_error.append(l.energy.pm / 1000)
			float_nuclide_level_energies.append(MeV_level_energy) # converts to MeV

		grid_levels = [] # renewed for each nuclide
		counter = 0
		for e in energy_grid: # target levels
			if counter > (len(float_nuclide_level_energies) - 1):
				grid_levels.append(np.nan)
			elif e < float_nuclide_level_energies[counter]:
				grid_levels.append(np.nan)
			else:
				grid_levels.append(float_nuclide_level_energies[counter])
				counter += 1

		for fl in grid_levels:
			full_level_energy_array.append(fl)

		full_level_energy_error_array.append(float_level_error)
		ns = temp_nuclide.mass - temp_nuclide.protons
		ensdf_nuclide_list.append([temp_nuclide.protons, ns])
	else:
		for i in energy_grid:
			full_level_energy_error_array.append(np.nan) # ERRORS
			full_level_energy_array.append(np.nan)
		print(endfb_nuclide, 'not in ENSDF')

fakedf = pd.DataFrame({'Z':full_Z_list, 'A':full_A_list, 't_levels': full_level_energy_array, 'c_levels':full_compound_level_list,
					   'ERG':full_energies_list, 'XS':full_xs_list,
					   'Q': full_q_list,
					   })
