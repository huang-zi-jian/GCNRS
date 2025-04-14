# GCNRS
output是MIA-GCF模型以及ADS-GSCL的实验结果日志。
scr为代码文件夹。MIA-GCF模型代码以MIA_SP_v8.py文件为准；ADS-GSCL模型代码以DRO_GCL为准。不同版本的模型代码文件对应着模型的消融变体以及一些深度模块分析。

### 1. 运行环境
在以下环境评估模型，并且使用Adam优化器。

```
Python version 3.9.12
torch==1.13.1
numpy==1.21.5
pandas=1.3.5
tqdm==4.64.1
```

### 2. 一些重要参数的说明

* `--topks` 生成推荐的K个物品列表。
* `--cl_weight` 是对比损失项 $\mathcal{L}_{cl}$的正则系数。
* `--str_weight` 是结构区域约束项 $\mathcal{L}_{S}$的正则系数。
* `--weight_decay` 是L2正则系数。
* `--temp` 对应对比损失函数中的温度系数 $\tau$。
* `--static_prob` 是交互边的丢弃比例。
* `--batch_size` 是训练批次数据量。
* `--embedding_dim` 是用户和物品的嵌入表示维度。
* `--n_layers` 是图神经网络的卷积层数。
