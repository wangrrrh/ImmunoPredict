# load data
import argparse
import scanpy as sc
import pandas as pd

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='处理单细胞数据')
parser.add_argument('--raw', type=str, help='是否使用原始数据')
parser.add_argument('--input_files', type=str, nargs='+', help='输入的文件路径')
parser.add_argument('--output_file', type=str, help='输出文件的路径')

args = parser.parse_args()

# 根据参数读取文件
input_files = args.input_files
output_path = args.output_file
# 根据 --raw 参数决定读取哪些文件
if args.raw.lower() == 'yes':
    if len(args.input_files) != 2:
        parser.error("当使用原始数据时，需要提供两个输入文件：矩阵数据文件(txt)和meta信息文件(csv)。")
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
    print(f'数据处理完毕，输出文件路径为：{output_path}')
else:
    file_path = input_files[0]
    adata = sc.read(file_path)
    print('数据读取完毕')


