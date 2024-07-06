from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal

import anndata as ad
import loompy as lp
import scanpy as sc
import numpy as np
import argparse
import scipy.sparse as sp
from datasets import Dataset

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
logger = logging.getLogger(__name__)

GENE_MEDIAN_FILE = "dict/gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = "dict/token_dictionary.pkl"

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='参数输入')
parser.add_argument('--input_file', type=str, help='输入文件的路径')
parser.add_argument('--output_file', type=str, help='输出文件的路径')

args = parser.parse_args()
input_file = args.input_file
output_path = args.output_file



def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    # 按照基因表达值降序排列
    sorted_indices = np.argsort(-gene_vector)
    sorted_indices = sorted_indices.astype(int)
    # 给出了按照基因表达值排序后，每个基因在该细胞中的标记顺序
    sort_gene = [gene_tokens[idx] for idx in sorted_indices]
    return sort_gene


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


class TranscriptomeTokenizer:
    def __init__(
            self,
            custom_attr_name_dict=None,
            nproc=30,
            chunk_size=512,  # 每次处理细胞的数量
            gene_median_file=GENE_MEDIAN_FILE,  # 基因中位数
            token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):

        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # chunk size for anndata tokenizer
        self.chunk_size = chunk_size

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_anndata(self, adata_file_path, target_sum=10_000):
        # 使用Path对象的as_posix()方法获取文件路径的字符串表示
        adata = sc.read(adata_file_path.as_posix())

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        # 存储了anndata中能在genelist_dict中找到的gene的位置编码
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
        )[0]
        # 存储了上面找到的基因在gene_median_dict的value值，即在30M数据集中的平均表达量
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in adata.var["ensembl_id"][coding_miRNA_loc]
            ]
        )

        # 找出来的这些基因对应的token值
        coding_miRNA_tokens = []
        genes_to_remove = []
        adata.var_names = [str(i) for i in range(len(adata.var_names))]
        for i in adata.var_names:
            ensg = adata.var["ensembl_id"][i]
            if adata.var["ensembl_id"][i] in self.gene_token_dict:
                coding_miRNA_tokens.append(self.gene_token_dict[ensg])
            else:
                # 处理基因 ID 不存在于字典中的情况，比如输出错误信息或添加默认值
                print(f"Gene ID {i} not found in gene_token_dict")
                genes_to_remove.append(i)
        adata.obs = adata.obs.iloc[:, 0:12]
        # 删除在 gene_token_dict 中不存在的基因对应的列
        adata = adata[:, ~adata.var_names.isin(genes_to_remove)]

        try:
            _ = adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True
        # 要不要筛细胞
        if var_exists:
            filter_pass_loc = np.where([i == 1 for i in adata.obs["filter_pass"]])[0]
        elif not var_exists:
            print(
                f"adata has no column attribute 'filter_pass'; tokenizing all cells."
            )
            # 这里的location是所有细胞，因为没有细胞被筛掉
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), self.chunk_size):
            # 将所有细胞分成一个一个chunk，然后每次循环获取一个块
            idx = filter_pass_loc[i: i + self.chunk_size]
            # 块中细胞的总计数
            n_counts = adata[idx].obs["n_counts"].values[:, None]
            # 块中每个细胞的基因表达数据
            X_view = adata[idx].X
            # X_norm = adata[idx].X
            # 对基因表达数据进行归一化
            X_norm = X_view / n_counts * target_sum / norm_factor_vector  # 前面的那个30M数据集的表达值均数在这里用上了
            X_norm = sp.csr_matrix(X_norm)

            # 对每一个cell进行tokenizer化
            # tokenized_cells是一个列表，其中的每个元素都是一个NumPy数组
            for i in range(X_norm.shape[0]):
                indices = X_norm[i].indices.astype(int)
                indexed_tokens = [coding_miRNA_tokens[idx] for idx in indices]
                tokenized_cells += [rank_genes(X_norm[i].data, indexed_tokens)]

            # add custom attributes for subview to dict
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def tokenize_data(
            self,
            data_directory: Path | str,
            output_directory: Path | str,
            output_prefix: str,
            file_format: Literal["loom", "h5ad"] = "h5ad",
            use_generator: bool = True,
    ):

        tokenized_cells, cell_metadata = self.tokenize_files(
            Path(data_directory), file_format
        )
        tokenized_dataset = self.create_dataset(
            tokenized_cells, cell_metadata, use_generator=use_generator
        )

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)

    def tokenize_files(
            self, data_directory, file_format: Literal["loom", "h5ad"] = "h5ad"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.values()
            }

        # loops through directories to tokenize .loom files
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )
        for file_path in data_directory.glob(f"*.{file_format}"):
            file_found = 1
            print(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[
                        k
                    ]
            else:
                cell_metadata = None

        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}."
            )
            raise
        return tokenized_cells, cell_metadata

    def tokenize_loom(self, loom_file_path, target_sum=10_000):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["ensembl_id"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                filter_pass_loc = np.where([i == 1 for i in data.ca["filter_pass"]])[0]
            elif not var_exists:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for _ix, _selection, view in data.scan(
                    items=filter_pass_loc, axis=1, batch_size=self.chunk_size
            ):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                        subview[:, :]
                        / subview.ca.n_counts
                        * target_sum
                        / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def create_dataset(
            self,
            tokenized_cells,
            cell_metadata,
            use_generator=False,
            keep_uncropped_input_ids=False,
    ):
        print("Creating dataset.")
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}

            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        def format_cell_features(example):
            # Store original uncropped input_ids in separate feature
            if keep_uncropped_input_ids:
                example["input_ids_uncropped"] = example["input_ids"]
                example["length_uncropped"] = len(example["input_ids"])

            # Truncate/Crop input_ids to size 2,048
            example["input_ids"] = example["input_ids"][0:2048]
            example["length"] = len(example["input_ids"])

            return example

        output_dataset_truncated = output_dataset.map(
            format_cell_features, num_proc=self.nproc
        )
        return output_dataset_truncated


def Sym2Ens(adata):
    with open("dict/gene_name_id_dict.pkl", 'rb') as f:
        gene_symbol_to_ensemble_id = pickle.load(f)
    genes_to_remove = []
    # 遍历adata中的vars.name，替换为ensembleID格式
    for i in range(len(adata.var_names)):
        gene_symbol = adata.var_names[i]
        gene_symbol_split = gene_symbol.split(".")[0]
        ensemble_id = gene_symbol_to_ensemble_id.get(gene_symbol_split)
        if ensemble_id is not None:
            adata.var["ensembl_id"][i] = ensemble_id
        else:
            print(gene_symbol)
            genes_to_remove.append(gene_symbol)

    # 删除无法转换的基因所对应的行
    adata = adata[:, ~adata.var_names.isin(genes_to_remove)]

    # 计算每个细胞的总计数
    n_counts_per_cell = np.sum(adata.X, axis=1)

    # 将计算得到的总计数添加到 AnnData 对象中
    adata.obs['n_counts'] = n_counts_per_cell
    return adata


adata = ad.read_h5ad(input_file)
adata.var['ensembl_id'] = adata.var_names
adata = Sym2Ens(adata)
sc.write('test_data/processed_data/GSE123813_ENSG_filtered.h5ad',adata)
adata.obs.isna()
custom_attributes = {
    'cell.id': 'cell.id',
    'patient': 'patient',  # 原始属性名映射到自身，或者您希望的新名称
    'treatment': 'treatment',
    'cluster': 'cluster',
    'response': 'response',
    'best_change': 'best_change'  # 如果您希望在tokenized数据中使用不同的属性名
}
tk = TranscriptomeTokenizer(custom_attributes, 16)
anndata_data_directory = "test_data/processed_data"
output_prefix = 'processed'
#
tk.tokenize_data(anndata_data_directory,
                 output_path,
                 output_prefix,
                 file_format="h5ad")
print("done")
