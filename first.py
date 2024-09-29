# -*- coding: utf-8 -*-
# load data
import argparse
import scanpy as sc
import pandas as pd

parser = argparse.ArgumentParser(description='Processing single-cell data')
parser.add_argument('--raw', type=str, help='Whether to use original data(yes/no)')
parser.add_argument('--input_files', type=str, nargs='+', help='Input file path')
parser.add_argument('--output_file', type=str, help='Output file path')

args = parser.parse_args()

input_files = args.input_files
output_path = args.output_file

if args.raw.lower() == 'yes':
    if len(args.input_files) != 2:
        parser.error("When using raw data, two input files need to be provided: matrix data file (txt) and meta information file (csv).")
    file_path_matrix = input_files[0]
    file_path_meta = input_files[1]
    df = pd.read_csv(file_path_matrix, delimiter='\t').T
    cellinfo = pd.DataFrame(df.index, index=df.index, columns=['sample_index'])
    geneinfo = pd.DataFrame(df.columns, index=df.columns, columns=['genes_index'])
    adata = sc.AnnData(df, obs=cellinfo, var=geneinfo)
    meta = pd.read_csv(file_path_meta,
                       delimiter=',', header=0)
    adata.obs = meta
    sc.write(output_path, adata)
    output_path = args.output_file
    print(f'Data processed，the ouput path is：{output_path}')
else:
    file_path = input_files[0]
    adata = sc.read(file_path)
    print('Data processed')


