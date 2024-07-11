import pandas as pd
import tqdm
import resml_functions
import numpy as np

df = pd.read_csv('ENDFBVIII_MT102_XS_only.csv')

Qs = pd.read_csv('ENDFBVIII_MT102_Q_values.csv')
Q_nucs = resml_functions.range_setter(df=Qs, la=0, ua=260)

full_q_list = []
for i, row in tqdm.tqdm(df.iterrows(), total = df.shape[0]):
	current_nuclide = [row['Z'], row['A']]

	reduced_q1 = Qs[Qs.Z == current_nuclide[0]]
	reduced_q2 = reduced_q1[reduced_q1.A == current_nuclide[1]]

	qval = reduced_q2['Q'].values[0]

	full_q_list.append(qval)
