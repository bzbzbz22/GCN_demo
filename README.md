### Can not obtain deterministic results when using `dgl.nn.pytorch.GraphConv`, even the random seeds are fixed.

In our experiment, we split the training/validation/test set for a graph data set (taking Cora as an example) many times. Each split was trained in a run and the classification accuracy was calculated.
We found that when we use 'dgl.nn.pytorch.GraphConv' to construct the GCN, although all random seeds (numpy, torch, dgl) are fixed, the experimental results of repeated experiments are still inconsistent.

Run `python main.py --use_dgl --dataset cora`, 
the results are:
```
Repeat 0:
Run 0, test_acc: 0.8451
Run 1, test_acc: 0.8571
Run 2, test_acc: 0.8270
Run 3, test_acc: 0.8290
Run 4, test_acc: 0.8290
Repeat 1:
Run 0, test_acc: 0.8592
Run 1, test_acc: 0.8551
Run 2, test_acc: 0.8149
Run 3, test_acc: 0.8229
Run 4, test_acc: 0.8209
```

Run `python main.py --use_dgl --dataset texas`, 
the results are:
```
Repeat 0:
Run 0, test_acc: 0.0541
Run 1, test_acc: 0.3784
Run 2, test_acc: 0.4054
Run 3, test_acc: 0.5135
Run 4, test_acc: 0.4595
Repeat 1:
Run 0, test_acc: 0.4054
Run 1, test_acc: 0.3784
Run 2, test_acc: 0.4054
Run 3, test_acc: 0.5405
Run 4, test_acc: 0.4324
```
As shown, for the two datasets: Cora and Texas, the experimental results of two repeats are inconsistent.

We also build another GCN that doesn't use `dgl.nn.pytorch.GraphConv `

Run `python main.py --dataset cora`, the results are:

```
Repeat 0:
Run 0, test_acc: 0.1650
Run 1, test_acc: 0.1288
Run 2, test_acc: 0.1429
Run 3, test_acc: 0.1469
Run 4, test_acc: 0.1368
Repeat 1:
Run 0, test_acc: 0.1650
Run 1, test_acc: 0.1288
Run 2, test_acc: 0.1429
Run 3, test_acc: 0.1469
Run 4, test_acc: 0.1368
```
Run `python main.py --dataset texas`, the results are:
```
Repeat 0:
Run 0, test_acc: 0.0541
Run 1, test_acc: 0.2973
Run 2, test_acc: 0.1351
Run 3, test_acc: 0.1081
Run 4, test_acc: 0.1081
Repeat 1:
Run 0, test_acc: 0.0541
Run 1, test_acc: 0.2973
Run 2, test_acc: 0.1351
Run 3, test_acc: 0.1081
Run 4, test_acc: 0.1081
```
As shown above, for GCN without  A. The experimental results of the two repeats are consistent. The poor accuracy of the model is due to the lack of normalization and other operations in the model, which do not affect whether the deterministic results are generated.

## Chinese
### 在使用  `dgl.nn.pytorch.GraphConv ` 时固定随机因子仍无法获得确定性结果

在我们的实验中，我们对一个图数据集(以cora为例)的训练/验证/测试集进行多次划分，每个划分进行一个run的训练并计算分类准确率结果
我们发现使用`dgl.nn.pytorch.GraphConv `构建的GCN在固定所有随机因子（numpy, torch，dgl）情况下，重复进行实验的实验结果仍不一致

执行`python main.py --use_dgl --dataset cora`, 
得到结果

```
Repeat 0:
Run 0, test_acc: 0.8451
Run 1, test_acc: 0.8571
Run 2, test_acc: 0.8270
Run 3, test_acc: 0.8290
Run 4, test_acc: 0.8290
Repeat 1:
Run 0, test_acc: 0.8592
Run 1, test_acc: 0.8551
Run 2, test_acc: 0.8149
Run 3, test_acc: 0.8229
Run 4, test_acc: 0.8209
```
执行`python main.py --use_dgl --dataset texas`, 
得到结果
```
Repeat 0:
Run 0, test_acc: 0.0541
Run 1, test_acc: 0.3784
Run 2, test_acc: 0.4054
Run 3, test_acc: 0.5135
Run 4, test_acc: 0.4595
Repeat 1:
Run 0, test_acc: 0.4054
Run 1, test_acc: 0.3784
Run 2, test_acc: 0.4054
Run 3, test_acc: 0.5405
Run 4, test_acc: 0.4324
```
可以看到，对于cora和texas数据集，两次Repeat的实验结果不一致。

我们再构建一个不使用`dgl.nn.pytorch.GraphConv `的GCN模型

执行`python main.py --dataset cora`, 得到结果

```
Repeat 0:
Run 0, test_acc: 0.1650
Run 1, test_acc: 0.1288
Run 2, test_acc: 0.1429
Run 3, test_acc: 0.1469
Run 4, test_acc: 0.1368
Repeat 1:
Run 0, test_acc: 0.1650
Run 1, test_acc: 0.1288
Run 2, test_acc: 0.1429
Run 3, test_acc: 0.1469
Run 4, test_acc: 0.1368
```
执行`python main.py --dataset texas`, 得到结果
```
Repeat 0:
Run 0, test_acc: 0.0541
Run 1, test_acc: 0.2973
Run 2, test_acc: 0.1351
Run 3, test_acc: 0.1081
Run 4, test_acc: 0.1081
Repeat 1:
Run 0, test_acc: 0.0541
Run 1, test_acc: 0.2973
Run 2, test_acc: 0.1351
Run 3, test_acc: 0.1081
Run 4, test_acc: 0.1081
```
如上所示，对于不使用`dgl.nn.pytorch.GraphConv `的GCN模型，两次Repeat的实验结果一致。该模型精度结果较差是由于该模型中没有归一化等操作，这并不影响是否产生确定性结果

