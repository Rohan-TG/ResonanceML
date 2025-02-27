# import pandas as pd
import h5py
import dask.dataframe


# df = pd.read_hdf('../AI_data/capture_xs_data_39.h5')

df = dask.dataframe.read_hdf('../AI_data/capture_xs_data_39.h5', key='xs_data')

print(df)

with h5py.File("../AI_data/capture_xs_data_39.h5", "r") as f:
	# dataset = f["my_data"]
	# print("Keys in the file:", list(f.keys()))
	# dataset = f["xs_data"][:]  # Read the entire dataset into memory
	print(f)