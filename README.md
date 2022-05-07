# nlp-tutorial
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/nlp-tutorial.svg)](https://github.com/shibing624/nlp-tutorial/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/nlp-tutorial.svg)](https://github.com/shibing624/nlp-tutorial/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

自然语言处理（NLP）教程，包括：文本词向量，词法分析，预训练语言模型，文本分类，文本语义相似度计算，文本生成，实体识别，翻译，对话。


在本NLP教程包含了一些范例，涵盖了大多数常见NLP任务，是入门NLP和PyTorch的学习资料，也可以作为工作中上手NLP的基线参考实现。


**Guide**

- [Tutorial](#nlp-tutorial的例子清单)
- [Get Started](#get-started)
- [Contact](#Contact)
- [Cite](#Cite)
- [Reference](#reference)

# nlp-tutorial的例子清单

- 目录说明

| **目录**   | **主题**          | 简要说明                             |
| ------------------------- | ------------------ | ----------------------------------- |
| [01_word_embedding](01_word_embedding)         | 词向量模型    | 提供了一个利用领域数据集提升词向量效果的例子。 |
| [02_lexical_analysis](02_lexical_analysis)       | 词法分析    | 词法分析任务的输入是一个句子，而输出是句子中的词边界和词性，这个例子基于LSTM和CRF实现。 |
| [03_language_model](03_language_model)         | 语言模型    | 提供了多个语言模型，如bert, electra, elmo, gpt等等，也提供了支持语言模型在垂直了类领域数据上继续训练的工具包。 |
| [04_text_classification](04_text_classification)    | 文本分类    | 使用机器学习和深度模型如何完成文本分类任务。 |
| [05_text_matching](05_text_matching)          | 文本匹配    | 提供了SBERT的文本匹配算法实现，可以应用于搜索，推荐系统排序，召回等场景。 |
| [06_text_generation](06_text_generation)        | 文本生成    | 包含BERT面向生成任务的预训练+微调模型框架，以及一个GPT模型的应用。 |
| [07_information_extraction](07_information_extraction) | 信息抽取    | 提供了实体识别的基线实现。包含人名、地址名称的命名实体识别。 |
| [08_machine_translation](08_machine_translation)    | 机器翻译    | 提供了一个带Attention机制的，基于LSTM的多层RNN Seq2Seq翻译模型。 |
| [09_dialogue](09_dialogue)               | 对话系统    | 提供了Bert的问答模型。|

- Notebook教程说明

| Notebook     |      Description      |    |
|:----------|:-------------|------:|
| [01_word_embedding/01_文本表示.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/01_word_embedding/01_文本表示.ipynb)  | 文本向量表示  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/01_word_embedding/01_文本表示.ipynb) |
| [01_word_embedding/02_词向量Skip-gram模型.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/01_word_embedding/02_词向量Skip-gram模型.ipynb)  | 实现skip-gram词向量模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/01_word_embedding/02_词向量Skip-gram模型.ipynb) |
| [01_word_embedding/03_Word2Vec.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/01_word_embedding/03_Word2Vec.ipynb)  | 基于gensim使用word2vec模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/01_word_embedding/03_Word2Vec.ipynb) |
| [01_word_embedding/04_Doc2Vec.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/01_word_embedding/04_Doc2Vec.ipynb)  | 基于gensim使用Doc2Vec模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/01_word_embedding/04_Doc2Vec.ipynb) |
| [01_word_embedding/05_Bert向量.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/01_word_embedding/05_Bert向量.ipynb)  | 使用Bert模型取文本向量  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/01_word_embedding/05_Bert向量.ipynb) |
| [02_lexical_analysis/01_中文分词工具.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/02_lexical_analysis/01_中文分词工具.ipynb)  | 中文分词工具  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/01_中文分词工具.ipynb) |
| [02_lexical_analysis/02_从头实现中文分词.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/02_从头实现中文分词.ipynb)  | 从头实现中文分词模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/02_从头实现中文分词.ipynb) |
| [02_lexical_analysis/03_LSTM词性标注模型.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/03_LSTM词性标注模型.ipynb)  | LSTM词性标注模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/03_LSTM词性标注模型.ipynb) |
| [03_language_model/01_语言模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/03_language_model/01_语言模型.ipynb)  | 从头实现RNN语言模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/02_lexical_analysis/01_语言模型.ipynb) |
| [03_language_model/02_Transformer语言模型.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/03_language_model/02_Transformer语言模型.ipynb)  | 从头实现Transformer语言模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/03_language_model/02_Transformer语言模型.ipynb) |
| [03_language_model/03_Bert完形填空.ipynb](https://github.com/shibing624/nlp-tutorial/blob/main/03_language_model/03_Bert完形填空.ipynb)  | 基于transformers使用Bert模型做完形填空  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/03_language_model/03_Bert完形填空.ipynb) |
| [04_text_classification/01_机器学习分类模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/04_text_classification/01_机器学习分类模型.ipynb)  | 基于scikit-learn训练LR等传统机器学习模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/04_text_classification/01_机器学习分类模型.ipynb) |
| [04_text_classification/02_深度学习分类模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/04_text_classification/02_深度学习分类模型.ipynb)  | 训练PyTorch的IMDb情感分析模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/04_text_classification/02_深度学习分类模型.ipynb) |
| [04_text_classification/03_Bert文本分类.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/04_text_classification/03_Bert文本分类.ipynb)  | 使用Bert模型finetune分类任务  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/04_text_classification/03_Bert文本分类.ipynb) |
| [04_text_classification/04_应用_姓名识别国籍.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/04_text_classification/04_应用_姓名识别国籍.ipynb)  | 从头实现RNN模型做人名的国籍分类  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/04_text_classification/04_应用_姓名识别国籍.ipynb) |
| [05_text_matching/01_词粒度文本匹配.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/05_text_matching/01_词粒度文本匹配.ipynb)  | 基于字面和word2vec的词文本匹配  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/05_text_matching/01_词粒度文本匹配.ipynb) |
| [05_text_matching/02_句粒度文本匹配.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/05_text_matching/02_句粒度文本匹配.ipynb)  | SentenceBert的句子相似度计算  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/05_text_matching/02_句粒度文本匹配.ipynb) |
| [05_text_matching/03_篇章粒度文本匹配.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/05_text_matching/03_篇章粒度文本匹配.ipynb)  | LDA主题提取做Doc相似度计算  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/05_text_matching/03_篇章粒度文本匹配.ipynb) |
| [06_text_generation/01_字符级人名生成.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/06_text_generation/01_字符级人名生成.ipynb)  | 从头实现Char-RNN做人名生成  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/06_text_generation/01_字符级人名生成.ipynb) |
| [06_text_generation/02_预训练文本生成模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/06_text_generation/02_预训练文本生成模型.ipynb)  | 基于transformers的GPT、XLNet生成模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/06_text_generation/02_预训练文本生成模型.ipynb) |
| [07_information_extraction/01_CRF实体识别模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/07_information_extraction/01_CRF实体识别模型.ipynb)  | 从头实现CRF实体识别模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/07_information_extraction/01_CRF实体识别模型.ipynb) |
| [07_information_extraction/02_Bert实体识别.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/07_information_extraction/02_Bert实体识别.ipynb)  | 基于transformers的Bert实体识别模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/07_information_extraction/02_Bert实体识别.ipynb) |
| [08_machine_translation/01_从头实现Seq2Seq模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/08_machine_translation/01_从头实现Seq2Seq模型.ipynb)  | 从头实现Seq2Seq翻译模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/08_machine_translation/01_从头实现Seq2Seq模型.ipynb) |
| [08_machine_translation/02_transformer翻译模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/08_machine_translation/02_transformer翻译模型.ipynb)  | 从头实现Transformer翻译模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/08_machine_translation/02_transformer翻译模型.ipynb) |
| [08_machine_translation/03_T5翻译模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/08_machine_translation/03_T5翻译模型.ipynb)  | T5翻译模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/08_machine_translation/03_T5翻译模型.ipynb) |
| [09_dialogue/01_对话模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/09_dialogue/01_对话模型.ipynb)  | 基于transformers的Bert问答模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/09_dialogue/01_对话模型.ipynb) |


# Get Started

- 使用Colab学习教程：

点击各教程的[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/01_word_embedding/01_文本表示.ipynb)即可在Google Colab打开Notebook并运行。

- 使用Jupyter Notebook学习教程：

1. 下载Python：建议使用Anaconda，Python环境和包一键装好，[Python3.7 版本](https://www.anaconda.com/products/individual)
2. 下载本项目：可以使用`git clone`，或者下载zip文件，解压到电脑
3. 打开Jupyter Notebook：打开终端，`cd`到本项目所在的文件夹，执行：```jupyter notebook ```，浏览器打开`01_word_embedding/01_文本表示.ipynb`，跟随教程交互使用


## 安装
#### 环境依赖
python >= 3.7

#### pip安装依赖
pip install -r requirements.txt

## 特色
本教程提供了多场景、多任务的NLP应用示例，基于PyTorch开发，动态库API简单易懂，调试方便，上手学习的同时可以用于生产研发。 

教程内容涵盖了NLP基础、NLP应用以及文本相关的拓展应用如文本信息抽取、对话模型等。

## 使用
打开各个子目录文件夹（如[01_word_embedding](01_word_embedding)）即可学习使用，各子目录的任务可独立运行，相互之间无依赖。

# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/nlp-tutorial.svg)](https://github.com/shibing624/nlp-tutorial/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624*，进Python-NLP交流群，备注：*姓名-公司名-NLP*
<img src="http://42.193.145.218/github_data/xm_wechat_erweima.png" width="200" />


# Cite

如果你在研究中使用了nlp-tutorial，请按如下格式引用：

```latex
@software{nlp-tutorial,
  author = {Xu Ming},
  title = {nlp-tutorial: NLP Tutorial for Beginners},
  year = {2021},
  url = {https://github.com/shibing624/nlp-tutorial},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加nlp-tutorial的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在本地进行单元测试
 - 确保所有单测都是通过的

之后即可提交PR。

# Reference

1. [nlp-in-python-tutorial](https://github.com/adashofdata/nlp-in-python-tutorial)
2. [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
