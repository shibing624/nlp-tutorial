# UnifiedTransformer

## 模型简介

近年来，人机对话系统受到了学术界和产业界的广泛关注并取得了不错的发展。开放域对话系统旨在建立一个开放域的多轮对话系统，使得机器可以流畅自然地与人进行语言交互，既可以进行日常问候类的闲聊，又可以完成特定功能，以使得开放域对话系统具有实际应用价值。具体的说，开放域对话可以继续拆分为支持不同功能的对话形式，例如对话式推荐，知识对话技术等，如何解决并有效融合以上多个技能面临诸多挑战。

[UnifiedTransformer](https://arxiv.org/abs/2006.16779)以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，采用灵活的注意力机制，十分适合对话生成任务。

本项目是UnifiedTransformer在 Paddle 2.0上的开源实现，包含了在[DuConv](https://www.aclweb.org/anthology/P19-1369/)数据集上微调和预测的代码。

## 快速开始

### 环境依赖

- sentencepiece
- termcolor

安装方式：`pip install sentencepiece termcolor`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── finetune.py # 模型finetune主程序入口
├── infer.py # 模型预测主程序入口
├── utils.py # 定义参数及一些工具函数
└── README.md # 文档说明
```

### 数据准备

**DuConv**是百度发布的基于知识图谱的主动聊天任务数据集，让机器根据构建的知识图谱进行主动聊天，使机器具备模拟人类用语言进行信息传递的能力。数据集的创新性是：强调了bot的主动性，并且在闲聊对话中引入了明确的对话目标，即将对话引导到特定实体上。数据中的知识信息来源于电影和娱乐人物领域有聊天价值的知识信息，如票房、导演、评价等，以三元组SPO的形式组织，对话目标中的话题为电影或娱乐人物实体。数据集中共有3万session，约12万轮对话，划分为训练集、开发集、测试集1和测试集2，其中测试集1中包含对话的response，而测试集2中只有对话历史。

为了方便用户快速测试，PaddleNLP Dataset API内置了DuConv数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds, test1_ds, test2_ds = load_dataset('duconv', splits=('train', 'dev', 'test_1', 'test_2'))
```

## Reference

- [UnifiedTransformer](https://arxiv.org/abs/2006.16779)
- [Knover/luge-dialogue](https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue)
- [DuConv](https://www.aclweb.org/anthology/P19-1369/)
- [dialogbot](https://github.com/shibing624/dialogbot)