# nlp-tutorial
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/nlp-tutorial.svg)](https://github.com/shibing624/nlp-tutorial/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/nlp-tutorial.svg)](https://github.com/shibing624/nlp-tutorial/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

NLP教程，包括：文本词向量，预训练语言模型，文本语义相似度计算，词法分析，信息抽取，实体关系抽取，文本分类，翻译，对话。


在本NLP教程包含了一些范例，涵盖了大多数常见NLP任务，是入门NLP和PyTorch的学习资料，也可以作为工作中上手NLP的基线参考实现。


**Guide**

- [Tutorial](#nlp-tutorial的例子清单)
- [Get Started](#get-started)
- [Contact](#Contact)
- [Cite](#Cite)
- [Reference](#reference)

# nlp-tutorial的例子清单

| **目录**   | **主题**          | 简要说明                             |
| ------------------------- | ------------------ | ----------------------------------- |
| [01_word_embedding](01_word_embedding)         | 词向量模型    | 提供了一个利用领域数据集提升词向量效果的例子。 |
| [02_lexical_analysis](02_lexical_analysis)       | 词法分析    | 词法分析任务的输入是一个句子，而输出是句子中的词边界和词性、实体类别，这个例子基于双向GRU和CRF实现。 |
| [03_language_model](03_language_model)         | 语言模型    | 提供了多个语言模型，如bert, electra, elmo, gpt等等，也提供了支持语言模型在垂直了类领域数据上继续训练的工具包。 |
| [04_text_classification](04_text_classification)    | 文本分类    | 使用机器学习和深度模型如何完成文本分类任务。 |
| [05_text_matching](05_text_matching)          | 文本匹配    | 提供了SBERT的文本匹配算法实现，可以应用于搜索，推荐系统排序，召回等场景。 |
| [06_text_generation](06_text_generation)        | 文本生成    | 包含BERT面向生成任务的预训练+微调模型框架，以及一个GPT模型的应用。 |
| [07_information_extraction](07_information_extraction) | 信息抽取    | 提供了多个数据集上的信息抽取基线实现。包含快递单信息抽取， MSRA-NER 数据集命名实体识别。 |
| [08_machine_translation](08_machine_translation)    | 机器翻译    | 提供了一个带Attention机制的，基于LSTM的多层RNN Seq2Seq翻译模型。 |
| [09_dialogue](09_dialogue)               | 对话系统    | 提供了 LIC 2021对话比赛基线, 开放域对话模型。|


# Get Started

## 安装
#### 环境依赖
python >= 3.6

#### pip安装
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

<img src="http://42.193.145.218/github_data/nlp_wechatgroup_erweima1.png" width="200" /><img src="http://42.193.145.218/github_data/xm_wechat_erweima.png" width="200" />


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
