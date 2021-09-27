# 文本分类任务

近年来随着深度学习的发展，模型参数的数量飞速增长。为了训练这些参数，需要更大的数据集来避免过拟合。然而，对于大部分NLP任务来说，构建大规模的标注数据集非常困难（成本过高），特别是对于句法和语义相关的任务。相比之下，大规模的未标注语料库的构建则相对容易。为了利用这些数据，我们可以先从其中学习到一个好的表示，再将这些表示应用到其他任务中。最近的研究表明，基于大规模未标注语料库的预训练模型（Pretrained Models, PTM) 在NLP任务上取得了很好的表现。

大量的研究表明基于大型语料库的预训练模型（Pretrained Models, PTM）可以学习通用的语言表示，有利于下游NLP任务，同时能够避免从零开始训练模型。随着计算能力的发展，深度模型的出现（即 Transformer）和训练技巧的增强使得 PTM 不断发展，由浅变深。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/327f44ff3ed24493adca5ddc4dc24bf61eebe67c84a6492f872406f464fde91e" width="60%" height="50%"> <br />
</p>

本图片来自于：https://github.com/thunlp/PLMpapers

本示例展示了如何以BERT([Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805))预训练模型Finetune完成多标签文本分类任务。


## 教程列表

- [01_机器学习分类模型.ipynb](01_机器学习分类模型.ipynb)
- [02_深度学习分类模型.ipynb](02_深度学习分类模型.ipynb)
- [03_Bert文本分类.ipynb](03_Bert文本分类.ipynb)
- [04_应用_姓名识别国籍.ipynb](04_应用_姓名识别国籍.ipynb)
- [AVG情感分析.py](sentiment_classification_avg.py)
- [RNN情感分析.py](sentiment_classification_rnn.py)
- [CNN情感分析.py](sentiment_classification_cnn.py)