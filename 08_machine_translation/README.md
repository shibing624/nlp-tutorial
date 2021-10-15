# 机器翻译

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：机器翻译，带Attention机制的翻译模型。Seq2Seq翻译模型，模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。


## 教程列表


| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [08_machine_translation/01_从头实现Seq2Seq模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/08_machine_translation/01_从头实现Seq2Seq模型.ipynb)  | 从头实现Seq2Seq翻译模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/08_machine_translation/01_从头实现Seq2Seq模型.ipynb) |
| [08_machine_translation/02_transformer翻译模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/08_machine_translation/02_transformer翻译模型.ipynb)  | 从头实现Transformer翻译模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/08_machine_translation/02_transformer翻译模型.ipynb) |
| [08_machine_translation/03_T5翻译模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/08_machine_translation/03_T5翻译模型.ipynb)  | T5翻译模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/08_machine_translation/03_T5翻译模型.ipynb) |
