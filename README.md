# GCNRS
实验结果的输出包括 MIA-GCF 模型以及 ADS-GSCL 模型的实验日志。相关代码文件均存储在 scr 文件夹中，其中 MIA-GCF 模型的实现以 MIA_SP_v8.py 文件为准，而 ADS-GSCL 模型的实现则以 DRO_GCL 文件为依据。不同版本的模型代码文件对应于模型的消融变体及各类深度模块分析，以便系统地评估和比较不同结构对模型性能的影响。

<br>
<p align='center'>
<img src="https://github.com/huang-zi-jian/GCNRS/blob/main/image/mia-gcf-arthitecture.png"  width="500" height="250"><br>
<i> (1): MIA-GCF的整体架构图 </i>
</p>

<br>
<p align='center'>
<img src="https://github.com/huang-zi-jian/GCNRS/blob/main/image/ads-secl-architecture.png"  width="500" height="200"><br>
<i> (2): ADS-GSCL的整体架构图 </i>
</p>

### 1. 运行环境
本地计算资源应至少满足以下配置：NVIDIA RTX 3060 GPU 和 16GB DDR4内存。这些配置能够有效支持大多数中小规模深度学习模型的训练和评估。然而，对于更复杂的模型，尤其是需要处理大规模数据集或涉及高维计算的任务，可能需要更强大的硬件资源才能确保模型训练和评估的顺利进行。在以下环境下进行模型评估，且采用 Adam优化器 来优化模型的训练过程。更详细的第三方库版本见requirements.txt文件。

```
Python version 3.9.12
torch==2.0.1
numpy==1.21.5
pandas=1.3.5
tqdm==4.65.0
```

### 2. 一些重要参数的说明

* `--topks` 生成推荐的K个物品列表。
* `--intent_number` 意图解耦数量 $X$。
* `--cl_weight` 是对比损失项 $\mathcal{L}_{cl}$的正则系数。
* `--str_weight` 是结构区域约束项 $\mathcal{L}_{S}$的正则系数。
* `--weight_decay` 是L2正则系数。
* `--temp` 对应对比损失函数中的温度系数 $\tau$。
* `--static_prob` 是交互边的丢弃比例。
* `--batch_size` 是训练批次数据量。
* `--embedding_dim` 是用户和物品的嵌入表示维度。
* `--n_layers` 是图神经网络的卷积层数。
