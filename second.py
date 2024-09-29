# -*- coding: utf-8 -*-
import argparse
import scanpy as sc
import pandas as pd
import numpy as np

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='参数输入')
parser.add_argument('--input_file', type=str, help='输入文件的路径')
parser.add_argument('--qc', type=str, help='是否进行质量控制')
parser.add_argument('--threshold', type=int, nargs='*', help='质量控制阈值选择')
parser.add_argument('--output_file', type=str, help='输出文件的路径')

args = parser.parse_args()
input_file = args.input_file
threshold = args.threshold
output_path = args.output_file
adata = sc.read_h5ad(input_file)
if args.qc.lower() == 'yes':
    # 按照Geneformer的数据处理阈值处理
    # 计算总读数并添加到 adata.obs 中
    adata.obs['total_reads'] = adata.X.sum(axis=1)
    # 保留了总读取计数在该数据集内均值的3个标准差范围内的细胞
    total_reads_mean = np.mean(adata.obs["total_reads"])
    total_reads_std = np.std(adata.obs["total_reads"])
    # 定义过滤条件一：总读数在其均值的3个标准差内
    if len(threshold) == 0:
        total_reads_lower_bound = total_reads_mean - 3 * total_reads_std
        total_reads_upper_bound = total_reads_mean + 3 * total_reads_std
    else:
        total_reads_lower_bound = total_reads_mean - threshold[0] * total_reads_std
        total_reads_upper_bound = total_reads_mean + threshold[0] * total_reads_std
    # 过滤数据集
    adata = adata[
        (adata.obs["total_reads"] >= total_reads_lower_bound) &
        (adata.obs["total_reads"] <= total_reads_upper_bound)
        ]
    # 过滤条件二：仅保留蛋白质编码基因与miRNA
    # 过滤线粒体DNA
    adata = adata[:, ~adata.var_names.str.match(r'^MT-')]
    # 过滤核糖体基因
    adata = adata[:, ~adata.var_names.str.match(r'^RP[SL0-9]')]
    # 过滤 ERCC 控制基因
    adata = adata[:, ~adata.var_names.str.match(r'^ERCC-')]
    # 过滤条件三：少于 7 个基因表达的细胞
    if len(threshold) == 0:
        sc.pp.filter_cells(adata, min_genes=7)
    else:
        sc.pp.filter_cells(adata, min_genes=threshold[1])
    # 过滤条件四：低质量细胞
    sc.write(output_path, adata)
    print("数据集质控完成，输出文件路径为：", output_path)

else:
    sc.write(output_path, adata)
    print("未进行数据集质控，输出文件路径为：", output_path)
