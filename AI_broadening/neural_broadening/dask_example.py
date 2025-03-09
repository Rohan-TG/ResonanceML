from __future__ import annotations

import dask.array as da
import h5py
import dask
import glob
import tensorflow as tf


def read_h5_dataset(file: str, dataset_key: str) -> da.Array:
	with h5py.File(file, "r") as f:
		dset = f[dataset_key]
		chunks = (1000, *dset.shape[1:])  # Choose appropriate chunk size
		return da.from_array(dset, chunks=chunks)


def load_and_prepocess_data(root_dir: str) -> da.Array:

	files = glob.glob("*.h5", root_dir=root_dir)
	print(f"{len(files)} files to load")


	datasets = [read_h5_dataset(f, "dataset") for f in files]
	data = da.concatenate(datasets, axis=0)

	print("Calculating mean and std")
	mean = data.mean().compute()
	std = data.std().compute()

	print("Normalising")
	data_normalised = (data - mean) / std

	print("Done")
	return data_normalised


def dask_to_tf_dataset(dask_array: da.Array, batch_size: int) -> tf.data.Dataset:
	def generator():
		for batch in dask_array.to_delayed():
			yield batch.compute()  # Convert delayed chunk to numpy
		
	return tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec(shape=(None, *dask_array.shape[1:]), dtype=tf.float32)).batch(batch_size)


def main():

	root_dir = "../h5data/dT_0.1K_200K_3500K/" # Replace with your data folder
	data_normalised = load_and_prepocess_data(root_dir)
	dataset = dask_to_tf_dataset(data_normalised, batch_size=32)


if __name__ == '__main__':
	main()
