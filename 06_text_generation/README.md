# 文本生成


## 教程列表


| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [06_text_generation/01_字符级人名生成.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/06_text_generation/01_字符级人名生成.ipynb)  | 从头训练Char-RNN做人名生成  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/06_text_generation/01_字符级人名生成.ipynb) |
| [06_text_generation/02_预训练文本生成模型.ipynb](https://github.com/shibing624/nlp-tutorial/tree/main/06_text_generation/02_预训练文本生成模型.ipynb)  | 基于transformers的GPT、XLNet生成模型  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/nlp-tutorial/blob/main/06_text_generation/02_预训练文本生成模型.ipynb) |

## 文本生成比赛简介

自然语言生成旨在让机器能够像人一样使用自然语言进行表达和交互，它是人工智能领域重要的前沿课题，近年来受到学术界和工业界广泛关注。

随着神经网络生成模型特别是预训练语言模型的迅速发展，机器生成文本的可读性和流畅性不断提升。然而，自动生成的文本中依然经常出现不符合原文或背景的错误事实描述，这种生成的事实一致性问题是自然语言生成进行落地应用的主要障碍之一，并逐渐受到研究学者的关注。

在[此比赛](https://aistudio.baidu.com/aistudio/competition/detail/105)中，我们将提供三个对事实一致性有较高要求的生成任务，包括文案生成(AdvertiseGen)、摘要生成(LCSTS_new)和问题生成(DuReaderQG)：

- 文案生成根据结构化的商品信息生成合适的广告文案；
- 摘要生成是为输入文档生成简洁且包含关键信息的简洁文本；
- 问题生成则是根据给定段落以及答案生成适合的问题。



