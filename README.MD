test_data获取方式：https://drive.google.com/drive/folders/1TQgtWMWT2Jmo_oaii2iyneMNix68oMlc?usp=sharing

# 功能点测试1：数据输入
## 原始单细胞转录组表达矩阵
`python first.py --raw yes --input_file test_data/rawdata/demo_data.txt test_data/rawdata/demo_metadata.csv --output_file test_data/demo.h5ad`
## 带有meta信息的H5AD格式数据
`python first.py --raw no --input_file test_data/demo.h5ad`

# 功能点测试2：参数选择
## 是否进行单细胞转录组表达矩阵质量控制
`python second.py --input_file test_data/demo.h5ad --qc yes --threshold --output_file test_data/demo_filtered.h5ad`
## 质量控制阈值选择
`python second.py --input_file test_data/demo.h5ad --qc yes --threshold 3 7 --output_file test_data/demo_filtered.h5ad`

# 功能点测试3：数据预处理
`python third.py --input_file test_data/demo_filtered.h5ad --output_file dataset`

# 功能点测试4：异常信息
## 输入数据异常
### 例如原始数据输入时缺少meta信息文件
`python first.py --raw yes --input_file test_data/rawdata/demo_data.txt --output_file test_data/demo.h5ad`

## 软件环境异常
```
conda env create -f environment.yml
conda activate test
python fourth.py
```

# 功能点测试5：统计结果
`python fifth.py --input_dataset test_data/dataset/demo.arrow --input_model model/ModelForInference --output_file result_pics`
