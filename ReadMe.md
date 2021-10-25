MGCN: Semi-supervised Classification in Multi-layer Graphs with Graph Convolutional Networks
====

Here is the code for node embedding in multi-layer networks with attributes written in Pytorch.
Ghorbani et.al. "MGCN: Semi-supervised Classification in Multi-layer Graphs with Graph Convolutional Networks" [1]



Usage 
------------
The main file is "train.py". It contains the running code for infra dataset with default parameters.
```python train.py```


Input Data
------------
You should create a folder in "data" folder, named "dataset_str" in the code. For example if your dataset_str is "infra", you should place files in the path "data/infra/". The folder contains the following information:
1) The adjacency list of within layer edges for every layer. Node numbering should be started from 0. The name of the files containing the within layer adjacency list should follow the pattern "dataset_name.adj#NumberOfLayer".
For example if your dataset name is "infra" with three layers, you should have "infra.adj0", "infra.adj1" and "infra.adj2" files that every one of them contains adjacency list like follow:
```
0 1
0 2
0 3
0 4
```
2) The adjacency list of between layer edges for every pair of layers (if available). The name of the files containing the between layer adjacency list should follow the pattern "dataset_name.bet#NumberOfLayer1_#NumberOfLayer2".
For example if your dataset name is "infra" with three layers, you should have "infra.bet0_1", "infra.bet0_2" and "infra.bet1_2" (One file for every pair of layers is enough)
3) Features and labels of every node in every layer. If features aren't available identity matrix should be used. The name of the files containing the features and labels should follow the pattern "dataset_name.feat#NumberOfLayer"
For example if your dataset name is "infra" with three layers, you should have "infra.feat0", "infra.feat1", "infra.feat2". Every line of file includes the node number, node features, node label. For example for "infra" dataset, if the first layer have 5 node belongs to 2 classes, it should be like follow:
```
NodeNumber Features Label
0 1 0 0 0 0 1
1 0 1 0 0 0 1
2 0 0 1 0 0 2
3 0 0 0 1 0 2
4 0 0 0 0 1 2
```

Parameters
------------
Parameters of the code are all in the list format for testing different combination of configs. 
Parameters are as follow:
- dataset_str: Dataset name
- adj_weights: It is a list for assigning different weights to input layers. If you leave it empty, all layers will have equal weihts. [[1,1,1]] includes one config with equal weights.
- wlambdas: It is the weight of reconstruction loss.
- hidden_structures: A list contains different structures for hidden spaces in every layer. For example [[[32],[32],[32]]] contains one config for hidden spaces of every layer which in all of them, the dimension of embedding for all layers are 32.  
- lrs: A list contains learning rate(s).
- test_sizes: A list contains the test size(s).

Note
------------
Thanks to Thomas Kipf. The code is written based on the "Graph Convolutional Networks in PyTorch" [2].

Bug Report
------------
If you find a bug, please send a bug report to mghorbani@ce.sharif.edu including if necessary the input file and the parameters that caused the bug.
You can also send me any comment or suggestion about the program.

References
------------
[1] [Ghorbani et.al., MGCN: Semi-supervised Classification in Multi-layer Graphs with Graph Convolutional Networks, 2019](https://arxiv.org/pdf/1811.08800)

[2] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

Cite
------------
Please cite our paper if you use this code in your own work:

```
@article{ghorbani2018multi,
  title={Multi-layered Graph Embedding with Graph Convolution Networks},
  author={Ghorbani, Mahsa and Baghshah, Mahdieh Soleymani and Rabiee, Hamid R},
  journal={arXiv preprint arXiv:1811.08800},
  year={2018}
}
```