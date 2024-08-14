import pandas as pd
import periodictable
import pandas as pd
import numpy as np
import resml_functions
import ENDF6
import tqdm
import os


# df = pd.read_csv('ENDFBVIII_MT102_XS_only.csv')




ENDF6_path = "/mnt/c/Users/TG300/ResonanceML/Data/TENDL2021/TENDL-n"

if os.path.exists(ENDF6_path) and os.path.isdir(ENDF6_path):
	files = os.listdir(ENDF6_path)

df = pd.DataFrame(columns=['Z', 'A', 'Q'])

for name in tqdm.tqdm(files, total=len(files)):
	with open(f"/mnt/c/Users/TG300/ResonanceML/Data/TENDL2021/TENDL-n/{str(name)}", 'r') as f:
		if len(name) == 10:
			element = periodictable.elements.symbol(name[2])
			nucleon_number = int(name[3:6])
			proton_number = element.number
		elif len(name) == 11:
			element = periodictable.elements.symbol(name[2:4])
			nucleon_number = int(name[4:7])
			proton_number = element.number

		# print(element, nucleon_number, proton_number)

		lines = f.readlines()

		try:
			sec = ENDF6.find_section(lines, MF=3, MT=103)
			x, y = ENDF6.read_table(sec)

			raw_q = sec[1].split()
			if raw_q[0][0] == '-':
				conjoined = raw_q[0]
				string_q = conjoined[:11]
			else:
				string_q = raw_q[0]
			converted_q = float(string_q[:-2] + 'e' + string_q[-1:])


			d = {'Z': proton_number,
				 'A': nucleon_number,
				 'Q' : converted_q}

			df = df._append(d, ignore_index=True)
		except:
			d = {'Z': proton_number,
				 'A': nucleon_number,
				 'Q' : np.nan}

			df = df._append(d, ignore_index=True)
			print(f"No MT103 data for {name}.")
			continue






