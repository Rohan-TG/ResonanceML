import pandas as pd
import tqdm
import resml_functions
import numpy as np

df = pd.read_csv('ENDFBVIII_MT102_XS_only.csv')

JENDL_Qs = pd.read_csv('JENDL5_MT102_Q_values.csv')
TENDL_Qs = pd.read_csv('TENDL_2021_MT102_Q_values.csv')
JENDL_Q_nucs = resml_functions.range_setter(df=JENDL_Qs, la=0, ua=260)

full_q_list = []
for i, row in tqdm.tqdm(df.iterrows(), total = df.shape[0]):

	current_nuclide = [row['Z'], row['A']]
	if current_nuclide in JENDL_Q_nucs:

		reduced_q1 = JENDL_Qs[JENDL_Qs.Z == current_nuclide[0]]
		reduced_q2 = reduced_q1[reduced_q1.A == current_nuclide[1]]

		qval = reduced_q2['Q'].values[0]

		full_q_list.append(qval)
	else:
		tendl_reducedq = TENDL_Qs[(TENDL_Qs.Z == current_nuclide[0]) & (TENDL_Qs.A == current_nuclide[1])]['Q'].values[0]
		full_q_list.append(tendl_reducedq)
