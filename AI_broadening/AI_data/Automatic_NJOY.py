# import os
# import numpy as np
import subprocess
import tqdm
import ENDF6
import pandas as pd

# temps = np.arange(300.0,1800.1,0.1)
# Temperatures = []
# for i in temps:
# 	Temperatures.append(round(i, 1))


Temperatures = [300.0, 2000.0]

deckTemplate = """moder
20 -21
reconr
-21 -22
'pendf Fe056 JENDL-5 '/
2631/
.001/
0/
moder
-22 36
broadr
-21 -22 -23
2631 1/
.001/
{broadTemp}/
0/
unresr
-21 -23 -24
2631 1 1 0/
{broadTemp}/
1.e+10/
0/
moder
-24 32
stop"""


for T in tqdm.tqdm(Temperatures, total=len(Temperatures)):
	deck = deckTemplate.format(broadTemp = T)
	deckName = f"deck_{T}_K.njoy"
	with open(f"deck_{T}_K.njoy", "w") as f:
		f.write(deck)

	commands = f"""
	ln -sf tape20 fort.20
	/home/rnt26/NJOY/NJOY21/bin/njoy21 < {deckName}
	ln -sf tape23 fort.23
	/home/rnt26/NJOY/NJOY21/bin/njoy21 < convert_to_ascii.njoy
	"""

	# t_output = subprocess.Popen(f"ln -sf tape20 fort.20 && /home/rnt26/NJOY/NJOY21/bin/njoy21 < {deckName} && ln -s tape23 fort.23 && /home/rnt26/NJOY/NJOY21/bin/njoy21 < convert_to_ascii.njoy")
	subprocess.run(commands, shell=True, executable="/bin/bash")
	filename = 'tape33'

	file = open(filename)

	lines = file.readlines()

	section = ENDF6.find_section(lines, MF=3, MT=102)

	x, y = ENDF6.read_table(section)

	df = pd.DataFrame({"ERG": x, "XS": y})
	df.to_csv(f'Fe56_{T}.csv')