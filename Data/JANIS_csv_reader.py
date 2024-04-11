import pandas as pd

df = pd.read_csv('table (1).csv',
				 # header=2,
				 delimiter=';',
				 )
df.columns = df.columns.str.split('.').str[0] # allows duplicate column names
old_column_names = df.columns

new_names = []
for name in old_column_names:
	new_names.append(name.strip())


parity_names = []
final_headers = []
for name in new_names:
	if name in parity_names:
		delta_header = name + ' ds'
		final_headers.append(delta_header)
	else:
		final_headers.append(name)
		parity_names.append(name)

df2 = df.set_axis(final_headers, axis=1)

# print(df.columns)