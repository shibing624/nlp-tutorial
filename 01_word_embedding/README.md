# Word Embedding


词向量（Word embedding），即把词语表示成实数向量。“好”的词向量能体现词语直接的相近关系。词向量已经被证明可以提高NLP任务的性能，例如语法分析和情感分析。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/54878855b1df42f9ab50b280d76906b1e0175f280b0f4a2193a542c72634a9bf" width="60%" height="50%"> <br />
</p>
<br><center>图1：词向量示意图</center></br>


## 加载TokenEmbedding

`TokenEmbedding()`参数
- `embedding_name`
将模型名称以参数形式传入TokenEmbedding，加载对应的模型。默认为`w2v.baidu_encyclopedia.target.word-word.dim300`的词向量。
- `unknown_token`
未知token的表示，默认为[UNK]。
- `unknown_token_vector`
未知token的向量表示，默认生成和embedding维数一致，数值均值为0的正态分布向量。
- `extended_vocab_path`
扩展词汇列表文件路径，词表格式为一行一个词。如引入扩展词汇列表，trainable=True。
- `trainable`
Embedding层是否可被训练。True表示Embedding可以更新参数，False为不可更新。默认为True。


### 认识一下Embedding
**`TokenEmbedding.search()`**
获得指定词汇的词向量。


### 词向量映射到低维空间

使用深度学习可视化工具[VisualDL](https://github.com/PaddlePaddle/VisualDL)的[High Dimensional](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md#High-Dimensional--%E6%95%B0%E6%8D%AE%E9%99%8D%E7%BB%B4%E7%BB%84%E4%BB%B6)组件可以对embedding结果进行可视化展示，便于对其直观分析，步骤如下：

1. 升级 VisualDL 最新版本。

`pip install --upgrade visualdl`

2. 创建LogWriter并将记录词向量。

3. 点击左侧面板中的可视化tab，选择‘token_hidi’作为文件并启动VisualDL可视化


#### 启动VisualDL查看词向量降维效果
启动步骤：
- 1、切换到「可视化」指定可视化日志
- 2、日志文件选择 'token_hidi'
- 3、点击「启动VisualDL」后点击「打开VisualDL」，选择「高维数据映射」，即可查看词表中前1000词UMAP方法下映射到三维空间的可视化结果:

![](https://user-images.githubusercontent.com/48054808/120594172-1fe02b00-c473-11eb-9df1-c0206b07e948.gif)

可以看出，语义相近的词在词向量空间中聚集(如数字、章节等)，说明预训练好的词向量有很好的文本表示能力。

使用VisualDL除可视化embedding结果外，还可以对标量、图片、音频等进行可视化，有效提升训练调参效率。关于VisualDL更多功能和详细介绍，可参考[VisualDL使用文档](https://github.com/PaddlePaddle/VisualDL/tree/develop/docs)。

## 基于TokenEmbedding衡量句子语义相似度

在许多实际应用场景（如文档检索系统）中， 需要衡量两个句子的语义相似程度。此时我们可以使用词袋模型（Bag of Words，简称BoW）计算句子的语义向量。

**首先**，将两个句子分别进行切词，并在TokenEmbedding中查找相应的单词词向量（word embdding）。

**然后**，根据词袋模型，将句子的word embedding叠加作为句子向量（sentence embedding）。

**最后**，计算两个句子向量的余弦相似度。

### 基于TokenEmbedding的词袋模型


使用`BoWEncoder`搭建一个BoW模型用于计算句子语义。

* `paddlenlp.TokenEmbedding`组建word-embedding层
* `paddlenlp.seq2vec.BoWEncoder`组建句子建模层


