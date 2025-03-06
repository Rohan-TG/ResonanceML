# import pandas as pd
import h5py
import os
import dask.dataframe
import pandas as pd
import tqdm


# df = pd.read_hdf('../AI_data/capture_xs_data_39.h5')

# Linux directory address
h5files = os.listdir('/home/rnt26/PycharmProjects/ResonanceML/AI_broadening/h5data/dT_0.1K_200K_3500K/')


xs_list = []
erg_list = []
t_list = []

filerange = range(0, len(h5files))
for f, num in zip(tqdm.tqdm(h5files, total = len(h5files)), filerange):
	df = dask.dataframe.read_hdf(f'../h5data/dT_0.1K_200K_3500K/{f}', key='xs_data')

	erg_series = df['ERG']
	erg_floats = erg_series.astype(float)
	erg_pandas = erg_floats.compute()

	xs_series = df['XS']
	xs_floats = xs_series.astype(float)
	xs_pandas = xs_floats.compute()

	t_series = df['T']
	t_floats = t_series.astype(float)
	t_pandas = t_floats.compute()

	dataframe = pd.DataFrame({'ERG': erg_pandas, 'XS': xs_pandas, 'T': t_pandas})
	dataframe.to_csv(f'dT_01K_200K_3500K_file_{num}.csv')








# with h5py.File("../AI_data/capture_xs_data_39.h5", "r") as f:
# 	# dataset = f["my_data"]
# 	# print("Keys in the file:", list(f.keys()))
# 	# dataset = f["xs_data"][:]  # Read the entire dataset into memory
# 	print(f)