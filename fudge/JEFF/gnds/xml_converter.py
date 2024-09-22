import subprocess
import os
import glob
import tqdm


directory = '/Users/rntg/PycharmProjects/ResonanceML/fudge/JEFF/gnds'

endf_filenames = glob.glob(os.path.join(directory, '*.endf'))

single_filenames = os.listdir(directory)
single_endf_filenames = [n for n in single_filenames if '.endf' in n]

for file in tqdm.tqdm(single_endf_filenames,total=len(single_endf_filenames)):
	xmlFileName = file.replace('.endf', '.xml')
	subprocess.Popen(
		f'source /Users/rntg/PycharmProjects/ResonanceML/fudge/fudge/bin/activate && endf2gnds.py {file} {xmlFileName} --skipBadData --continuumSpectraFix',
		shell=True, text=True)



