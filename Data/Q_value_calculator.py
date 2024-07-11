import pandas as pd
import periodictable
import pandas as pd
import numpy as np
import resml_functions
import ENDF6
import tqdm
import os


df = pd.read_csv('ENDFBVIII_MT102_XS_only.csv')




ENDF6_path = "/mnt/c/Users/TG300/ResonanceML/Data/endftables_data/ENDF-B-VIII.0_neutrons/ENDF-B-VIII.0_neutrons"

if os.path.exists(ENDF6_path) and os.path.isdir(ENDF6_path):
	files = os.listdir(ENDF6_path)

df = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])

for name in tqdm.tqdm(files, total=len(files)):
	with open(f"/mnt/c/Users/TG300/ResonanceML/Data/endftables_data/ENDF-B-VIII.0_neutrons/ENDF-B-VIII.0_neutrons/{str(name)}", 'r') as f:
		if len(name) == 15:
			element = periodictable.elements.symbol(name[6])
			nucleon_number = int(name[2:5])
			proton_number = element.number
		elif len(name) == 16:
			element = periodictable.elements.symbol(name[6:8])
			nucleon_number = int(name[2:5])
			proton_number = element.number

		print(element, nucleon_number, proton_number)

		lines = f.readlines()

		try:
			sec = ENDF6.find_section(lines, MF=3, MT=16)
			x, y = ENDF6.read_table(sec)
			x = [i / 1e6 for i in x]

			t_ERG = []
			t_XS = []

			for ix, iy in zip(x, y):
				if ix < 20.0:
					d = {'A': nucleon_number,
						 'Z': proton_number,
						 'XS': iy,
						 'ERG': ix}
		except:
			print(f"No MT16 data for {name}")
			continue






