import pandas as pd
from nudel.nudel import Nuclide
import tqdm
from resml_functions import range_setter, General_plotter

maindf = pd.read_csv('ENDFBVIII_MT102_XS_Q_ZA.csv')


ENDFB_nuclides = range_setter(maindf, 0, 260, True)

energy_grid, unused = General_plotter(maindf, nuclides = [[17,35]])

print('Data loaded')
#
# full_level_energy_array = []
# full_level_energy_error_array = []
ensdf_nuclide_list = []

# eV_compound_list = []

compound_level_list = []
compound_level_error_list = []


Z_list = []
A_list = []


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

clevel_dataframe = pd.DataFrame(columns=['Z', 'A', 'c_levels', 'err_c_levels'])

for n in tqdm.tqdm(ENDFB_nuclides, total=len(ENDFB_nuclides)):
	compound = [n[0], n[1]+1]


	if compound not in missing_compounds:

		compound_nuclide = Nuclide(compound[1], compound[0])

		compound_level_structure = compound_nuclide.adopted_levels.levels

		for lc in compound_level_structure: # loop gives the levels of the compound nucleus in MeV
			lc_energy = lc.energy.val
			eV_comp_level_energy = lc_energy * 1e3 # in eV


			Z_list.append(n[0])
			A_list.append(n[1])
			compound_level_list.append(eV_comp_level_energy)
			compound_level_error_list.append(lc.energy.pm*1e3)
	else:
		print(f'No compound data for target {n}')
