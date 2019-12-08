# Blockflow: An implementation of workflow for optimizable modules

Hi friends, we are here together to build a workflow, as a contract, to collaborate more efficiently. 

## INSTALLATION

At this time, Blockflow only support `tensorflow`, however, we are going to make it available on Pytorch in the near future.

First, we have to clone the `datahub` repository first: https://gitlab.com/dinosaurchi/datahub

This library consists of the functions helping to build the Blockflow basis.


## DATA PREPARATION

Please find the standard dataset for each project at here: https://cinnamon-lab.app.box.com/folder/86814136054

Then, download the interesting one and run the code.


## DEMO

#### 1. Fitting:

Suppose that you want to play with `Okaya` dataset:

```
python demo\train.py --data data\okaya\samples\ --corpus data\okaya\corpus.json --classes data\okaya\classes.json --train data\okaya\train-qa-ocr.lst --val data\okaya\val-qa-ocr.lst --res report\testing
```

Please run the following command for descriptions of arguments:
```
python demo\train.py --h
```

#### 2. Freezing

Suppose that you want to pack a trained model for `Okaya` dataset, and given that the checkpoint is at `report\testing_2\checkpoints\checkpoint_epoch_10\`:

```
python utils\freeze_block.py --path report\testing_2\checkpoints\checkpoint_epoch_10\ --outmap freeze_maps\HeuristicGraph.json
```

Note that the `freeze_maps\HeuristicGraph.json` node mapping file indicates the name of the nodes should be changed (for normalization) and specify the `output` of the computational graph (the subgraph to be frozen). 

The frozen model file will be at `report\testing_2\checkpoints\checkpoint_epoch_10\frozen_model.pb`.

Please run the following command for descriptions of arguments:
```
python utils\freeze_block.py --h
```

#### 3. Inferring:

Suppose that you want to infer a trained model for `Okaya` dataset, and the pre-trained frozen model is at `report\testing_2\checkpoints\checkpoint_epoch_10\frozen_model.pb`:

```
python demo\infer.py --data data\okaya\samples\ --corpus data\okaya\corpus.json --classes data\okaya\classes.json --pb report\testing_2\checkpoints\checkpoint_epoch_10\frozen_model.pb
```

The resultant directory and the statistical information can be found in `inference` directory, with a specific running time stamp.

Please run the following command for descriptions of arguments:
```
python demo\infer.py --h
```