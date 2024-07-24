import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import periodictable
import math
from resml_functions import General_plotter

df = pd.read_csv('ENDFBVIII_resonance_parameter_trial_1.csv')


def p(nuclide, llim=0, ulim=61506):
	erg, xs = General_plotter(df, nuclides=[nuclide])
	minidf = df[(df['Z'] == nuclide[0]) & (df['A'] == nuclide[1])]
	params = minidf['res_e']
	# loge = [np.log10(e) for e in erg]
	# logxs = [np.log10(s) for s in xs]
	paramrep = []
	for idx, k in enumerate(params):
		if not math.isnan(k):
			paramrep.append(xs[idx])
		else:
			paramrep.append(np.nan)

	plt.figure()
	plt.plot(erg[llim:ulim], xs[llim:ulim], label='ENDF/B-VIII $\sigma_{n,\gamma}$')
	plt.plot(erg[llim:ulim], paramrep[llim:ulim], 'x', alpha=0.5,label='Resonance energy parameters')
	plt.xscale('log')
	plt.title(f'Resonance energy representation for {periodictable.elements[nuclide[0]]}-{nuclide[1]}')
	plt.yscale('log')
	plt.xlabel('Energy / eV')
	plt.ylabel('$\sigma_{n,\gamma}$ / b')
	plt.grid()
	plt.legend()
	plt.show()