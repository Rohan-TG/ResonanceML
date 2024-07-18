import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from resml_functions import range_setter, General_plotter
import numpy as np
from Data.ENSDF.nudel.nudel import Nuclide
import math

df = pd.read_csv('ENDFBVIII_resonance_parameter_trial_1.csv')
nuc = [6,12]
erg, xs = General_plotter(df=df, nuclides=[nuc])

rps = df[(df['Z'] == nuc[0]) & (df['A'] == nuc[1])]['res_e'].values

def fullplotter(lim1=0, lim2=67000):

	loge = [np.log10(e) for e in erg]
	logxs = [np.log10(x) for x in xs]

	logrp = []
	for i in rps:
		if not math.isnan(i):
			logrp.append(0)
		else:
			logrp.append(np.nan)

	plt.figure()
	plt.plot(loge[lim1:lim2],logxs[lim1:lim2])
	plt.grid()
	plt.plot(loge[lim1:lim2], logrp[lim1:lim2])
	plt.show()

al = range_setter(df, la=0, ua=260, use_tqdm=True)
maxes = []
for nuclide in tqdm.tqdm(al, total=len(al)):
	comp = [nuclide[0], nuclide[1]+1]
	qval = df[(df['Z'] == nuclide[0]) & (df['A'] == nuclide[1])]['Q'].values[0]
	try:
		nudel_extract = Nuclide(comp[1], comp[0])
		levels = nudel_extract.adopted_levels.levels
		ev_levels = [l.energy.val * 1e3 for l in levels]
		maxlevel = max(ev_levels)
		if maxlevel < qval:
			maxes.append(nuclide)
	except:
		print('No compound data for ', nuclide)
