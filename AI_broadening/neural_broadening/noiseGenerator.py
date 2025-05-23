import os
import random
import numpy as np

def NoiseXS(inputXS, stdScale):
	"""stdScale: fraction of real std used to generate noise, e.g. 0.1 is 10% of actual std of the value
	inputXS: all cross sections for a single temperature"""
	realstd = np.std(inputXS)

	noisyXS = []
	for xs in inputXS:
		noise = random.gauss(mu=0, sigma=stdScale * realstd)
		noisyXS.append(xs + noise)

	return noisyXS

def NoiseTemp(inputTemp, stdVal):
	"""stdScale: fraction of real std used to generate noise, e.g. 0.1 is 10% of actual std of the value
	inputTemp: array of the temperatures for a single sample (i.e. single valued)"""

	noise = random.gauss(mu=0, sigma=stdVal)
	noisyTemps = [T + noise for T in inputTemp]

	return noisyTemps