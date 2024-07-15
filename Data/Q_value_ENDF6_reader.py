import pandas as pd
import periodictable
import pandas as pd
import numpy as np
import resml_functions
import ENDF6
import tqdm
import os


# df = pd.read_csv('ENDFBVIII_MT102_XS_only.csv')




ENDF6_path = "/mnt/c/Users/TG300/ResonanceML/Data/JENDL5_u20/jendl5-n-u20/jendl5-n-u20"

if os.path.exists(ENDF6_path) and os.path.isdir(ENDF6_path):
	files = os.listdir(ENDF6_path)

df = pd.DataFrame(columns=['Z', 'A', 'Q'])

for name in tqdm.tqdm(files, total=len(files)):
	with open(f"/mnt/c/Users/TG300/ResonanceML/Data/JENDL5_u20/jendl5-n-u20/jendl5-n-u20/{str(name)}", 'r') as f:
		if len(name) == 19:
			element = periodictable.elements.symbol(name[6])
			nucleon_number = int(name[8:11])
			proton_number = element.number
		elif len(name) == 20:
			element = periodictable.elements.symbol(name[6:8])
			nucleon_number = int(name[9:12])
			proton_number = element.number

		# print(element, nucleon_number, proton_number)

		lines = f.readlines()

		try:
			sec = ENDF6.find_section(lines, MF=3, MT=102)
			x, y = ENDF6.read_table(sec)

			raw_q = sec[1].split()
			string_q = raw_q[0]
			converted_q = float(string_q[:-2] + 'e' + string_q[-1:])


			d = {'Z': proton_number,
				 'A': nucleon_number,
				 'Q' : converted_q}

			df = df._append(d, ignore_index=True)
		except:
			print(f"No MT102 data for {name}")
			continue






