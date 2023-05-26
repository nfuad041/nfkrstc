#!/usr/bin/env python3
import argparse
import tables as pt #PyTable
import pandas as pd
import pdb

#Command on bash: python *path to read_hdf5.py* *path to hdf5 file*

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Name of the  hdf5 file')
parser.add_argument('is_pandas_Dataframe', help='is pandas Dataframe (True/False)? Default False', default=False) 
args = parser.parse_args()

#example dsp file: /global/cfs/cdirs/m2676/data/krstc/LH5/dsp/krstc_run1_cyc2038_dsp.lh5

#example raw file: /global/cfs/cdirs/m2676/data/krstc/LH5/raw/krstc_run1_cyc2038_raw.lh5

head = 20

def read_file(filename):
	file = pt.open_file(filename)
	print(file)
	while True:
		print()
		x = input('path/to/a/dataset or (E)xit: ')
		print()
		if x=='E' or x=='e' or x=='Exit' or x=='exit':
			exit()
		else:
			print('dtype: ',end='')
			exec('print(file.root' + x.replace('/','.')+'.dtype)')
			for i in range(head):
				print(str(i)+'  ',end='')
				exec('print(file.root' + x.replace('/','.')+'['+ str(i)+'])')

def read_pandas_dataframe(filename):
	file = pd.read_hdf(filename)
	print(file.columns)
	print(file)
	while True:
		print()
		x = input('Column name or (E)xit: ')
		print()
		if x=='E' or x=='e' or x=='Exit' or x=='exit':
			exit()
		else:
			print(file[x])
	

if __name__ == '__main__':
	if(args.is_pandas_Dataframe==True):
		read_pandas_dataframe(args.filename)
	else:
		read_file(args.filename)
	
	
	
	
	
