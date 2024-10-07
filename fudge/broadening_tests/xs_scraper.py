import pandas as pd
import numpy as np
import os
import subprocess
import periodictable

isotope = [27, 59]

processed_GNDS_name = f'{isotope[0]}-{periodictable.elements[isotope[0]]}-{isotope[1]}g.proc.xml'
processed_GNDS_master_directory = f'/Users/rntg/PycharmProjects/ResonanceML/fudge/heated_gnds'

master_directory = f'/Users/rntg/PycharmProjects/ResonanceML/fudge/broadening_tests/n+{periodictable.elements[isotope[0]]}{isotope[1]}'

index_address = f'{master_directory}/index'
with open(index_address, 'r') as index_file:
	lines = index_file.readlines()
	for l in lines[2:]:
		SP = l.split()
		if SP[1] == '102':
			capture_index = SP[0]
			break




t_output = subprocess.Popen(f'source /Users/rntg/PycharmProjects/ResonanceML/fudge/fudge/bin/activate && temperatures.py {processed_GNDS_master_directory}/{processed_GNDS_name}',
							shell=True,
							text=True,
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE)

stdout, stderr = t_output.communicate()

list_stdout = stdout.split(r'\n')
full_split_stdout = []
for ITEM in list_stdout:
	result = ITEM.split()
	for x in result:
		full_split_stdout.append(x)

temps = []
procs_names = []
for i, item in enumerate(full_split_stdout):
	if 'heated_' in item and len(item) <15:
		temp = full_split_stdout[i-1]
		proc_name = item
		temps.append(temp)
		procs_names.append(proc_name)

master_heated_directory = f'{master_directory}/heated'

heated_dirs = os.listdir(master_heated_directory)
heated_dirs.remove('.DS_Store') # may need to run if on Mac

for heatedFolder in heated_dirs:
	T_index = procs_names.index(heatedFolder)
	associatedTemperature = temps[T_index]
