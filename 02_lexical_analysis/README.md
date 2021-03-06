# 词法分析（Lexical Analysis）


## 简介

词法分析任务的输入是一个字符串（我们后面使用『句子』来指代它），而输出是句子中的词边界和词性、实体类别。序列标注是词法分析的经典建模方式，我们使用基于 GRU 的网络结构学习特征，将学习到的特征接入 CRF 解码层完成序列标注。模型结构如下所示：<br />

![GRU-CRF-MODEL](https://paddlenlp.bj.bcebos.com/imgs/gru-crf-model.png)

1. 输入采用 one-hot 方式表示，每个字以一个 id 表示
2. one-hot 序列通过字表，转换为实向量表示的字向量序列；
3. 字向量序列作为双向 GRU 的输入，学习输入序列的特征表示，得到新的特性表示序列，我们堆叠了两层双向 GRU 以增加学习能力；
4. CRF 以 GRU 学习到的特征为输入，以标记序列为监督信号，实现序列标注。


## 教程列表

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [02_lexical_analysis/01_中文分词工具.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/02_lexical_analysis/01_中文分词工具.ipynb)  | 中文分词工具  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/01_中文分词工具.ipynb) |
| [02_lexical_analysis/02_从头实现中文分词.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/02_实体识别.ipynb)  | 从头实现中文分词模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/02_从头实现中文分词.ipynb) |
| [02_lexical_analysis/03_LSTM词性标注模型.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/03_LSTM词性标注模型.ipynb)  | LSTM词性标注模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/03_LSTM词性标注模型.ipynb) |
