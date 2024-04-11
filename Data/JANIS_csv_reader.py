import pandas as pd
import periodictable

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

def nuclide_extractor(head):
	element_name = ''
	isotope_string = ''
	for char in header:
		if char == ' ':
			break
		try:
			integer = int(char)
			isotope_string += char
		except ValueError:
			element_name += char

	isotope = int(isotope_string)
	atomic_number = periodictable.elements.symbol(element_name).number
	print(atomic_number, isotope)

for header in df2.columns[1:]:
	nuclide_extractor(head=header)





mt102 = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])


ERG = []
for i, row in df2.iterrows():
	# print(row)
	if i > 1:
		energy = float(row['Incident energy'].strip())
		if energy
		ERG.append(float(row['Incident energy'].strip()))





