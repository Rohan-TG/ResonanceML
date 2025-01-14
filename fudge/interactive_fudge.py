import os
import glob

directory = '/Users/rntg/PycharmProjects/ResonanceML/fudge/JEFF'

files = glob.glob(os.path.join(directory, '*.jeff33'))

for file in files:
	# Create the new filename by replacing .jeff33 with .endf
	new_file = file.replace('.jeff33', '.endf')

	# Rename the file
	os.rename(file, new_file)
	print(f'Renamed: {file} -> {new_file}')